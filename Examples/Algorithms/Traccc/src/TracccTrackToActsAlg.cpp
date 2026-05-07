// This file is part of the Acts project.
//
// Copyright (C) 2024 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Traccc/TracccTrackToActsAlg.hpp"

#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/EventData/VectorTrackContainer.hpp"
#include "ActsPlugins/Traccc/TrackConversion.hpp"
#include "ActsExamples/Framework/AlgorithmContext.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "Acts/EventData/SourceLink.hpp"

#include <stdexcept>

#include <traccc/edm/track_container.hpp>

namespace ActsExamples {

TracccTrackToActsAlg::TracccTrackToActsAlg(
    const Config& cfg, std::unique_ptr<const Acts::Logger> logger)
    : IAlgorithm("TracccTrackToActsAlg", std::move(logger)), m_cfg(cfg) {

  if (!m_cfg.trackingGeometry) {
    throw std::invalid_argument(
        "TracccTrackToActsAlg: trackingGeometry is null");
  }
  if (m_cfg.inputTracccTracks.empty()) {
    throw std::invalid_argument(
        "TracccTrackToActsAlg: inputTracccTracks is empty");
  }

  m_surfaceMap =
      Acts::TracccPlugin::buildSurfaceMap(*m_cfg.trackingGeometry);

  m_outputActsTracks.initialize(m_cfg.outputActsTracks);
  m_inputTracccTracks.initialize(m_cfg.inputTracccTracks);
  m_inputDetrayToActsMap.initialize(m_cfg.inputDetrayToActsMap);

}

// Convert a traccc host track container into a mutable Acts TrackContainer.
template <typename track_container_t,
          typename trajectory_t,
          template <typename> typename holder_t>
std::size_t convertTracks(
    traccc::edm::track_container<traccc::default_algebra>::const_view tracccTracksView,
    Acts::TrackContainer<track_container_t, trajectory_t, holder_t>& outputTracks,
    const std::unordered_map<std::uint64_t, Acts::GeometryIdentifier>& detrayToActsMap,
    const std::unordered_map<Acts::GeometryIdentifier, const Acts::Surface*>& surfaceMap) {

  traccc::edm::track_container<traccc::default_algebra>::const_device
      tracks_device(tracccTracksView);

  const auto& tracks       = tracks_device.tracks;
  const auto& states       = tracks_device.states;
  const auto& measurements = tracks_device.measurements;

  std::size_t nAccepted = 0;
  constexpr Acts::TrackStatePropMask kSmoothedMask =
      Acts::TrackStatePropMask::Smoothed |
      Acts::TrackStatePropMask::Calibrated;

  // In convertTracks, before the track loop, print a few map entries to compare
  std::size_t nPrinted = 0;
  for (const auto& [detrayId, actsId] : detrayToActsMap) {
      if (nPrinted++ > 3) break;
      auto it = surfaceMap.find(actsId);
      std::cerr << "detray=" << detrayId
                << " actsId=" << actsId
                << " found=" << (it != surfaceMap.end()) << "\n";
  }

  // Print the first few fit outcomes as raw integers:
  for (std::size_t i = 0; i < tracks.size(); ++i) {
      if (i > 5) break;
      auto outcome = tracks.at(i).fit_outcome();
      std::cerr << "track " << i
                << " fit_outcome=" << static_cast<int>(outcome)
                << " ndf=" << tracks.at(i).ndf() << "\n";
  }

  std::size_t nRejFitOutcome = 0, nRejMinMeas = 0, nRejNdf = 0, nRejState = 0;
  for (std::size_t iTrk = 0; iTrk < tracks.size(); ++iTrk) {
    const auto& track = tracks.at(iTrk);

    const auto fitOutcome = track.fit_outcome();

    if (track.fit_outcome() != traccc::track_fit_outcome::SUCCESS) {
        nRejFitOutcome++;
        continue;
    }

    if (track.constituent_links().size() < 4) {
      nRejMinMeas++;
      continue;
    }

    const float ndf = track.ndf();
    if (ndf >
            static_cast<float>(std::numeric_limits<unsigned int>::max()) ||
        ndf <
            static_cast<float>(std::numeric_limits<unsigned int>::min())) {
        nRejNdf++;
        continue;
    }

    auto actsTrack = outputTracks.makeTrack();
    actsTrack.chi2() = track.chi2();
    actsTrack.nDoF() = static_cast<unsigned int>(ndf);
    actsTrack.nMeasurements() = static_cast<std::uint32_t>(
        track.constituent_links().size());
    actsTrack.nHoles() = 0;
    actsTrack.nOutliers() = 0;

    bool firstState = true;
    bool trackOk    = true;

    for (const auto& [linkType, stateIdx] : track.constituent_links()) {
      assert(linkType == traccc::edm::track_constituent_link::track_state);

      const auto& state = states.at(stateIdx);
      const auto& meas = measurements[state.measurement_index()];


      auto optParams =
          Acts::TracccPlugin::convertState(state, meas, detrayToActsMap, surfaceMap);
      if (!optParams.has_value()) {
        trackOk = false;
        break;
      }

      if (firstState) {
        const std::uint64_t detrayId =
            state.smoothed_params().surface_link().value();
        auto optGlobal = Acts::TracccPlugin::convertGlobalParams(
            detrayId, track.params(), detrayToActsMap, surfaceMap);
        if (!optGlobal.has_value()) {
          trackOk = false;
          break;
        }
        actsTrack.parameters() = optGlobal->parameters();
        actsTrack.covariance() = optGlobal->covariance().value();
        actsTrack.setReferenceSurface(
            optGlobal->referenceSurface().getSharedPtr());
        firstState = false;
      }

      auto tsos = actsTrack.appendTrackState(kSmoothedMask);
      tsos.setReferenceSurface(optParams->referenceSurface().getSharedPtr());
      tsos.smoothed()           = optParams->parameters();
      tsos.smoothedCovariance() = optParams->covariance().value();
      tsos.typeFlags().setHasMeasurement(true);
      ActsExamples::IndexSourceLink sl(optParams->referenceSurface().geometryId(), meas.identifier());
      tsos.setUncalibratedSourceLink(Acts::SourceLink{sl}); // store original Acts measurement index in source link

    }

    if (!trackOk) {
      outputTracks.removeTrack(actsTrack.index());
      nRejState++;
      continue;
    }

    ++nAccepted;
  }

  std::cerr << "convertTracks: total=" << tracks.size()
            << " rejFitOutcome=" << nRejFitOutcome
            << " rejMinMeas=" << nRejMinMeas
            << " rejNdf=" << nRejNdf
            << " rejState=" << nRejState
            << " accepted=" << nAccepted << "\n";

  return nAccepted;
}

ProcessCode TracccTrackToActsAlg::execute(const AlgorithmContext& ctx) const {

    const auto& tracccTracks = m_inputTracccTracks(ctx);
    const auto& detrayToActsMap = m_inputDetrayToActsMap(ctx);
    // const auto& actsToTracccIndexMap = m_inputActsToTracccIndexMap(ctx);

    // if (!actsToTracccIndexMap.empty()) {
    //     ACTS_INFO("TracccTrackToActsAlg: got actsToTracccIndexMap with "
    //              << actsToTracccIndexMap.size() << " entries");
    // }

    auto trackBackend    = std::make_shared<Acts::VectorTrackContainer>();
    auto trajectoryBackend = std::make_shared<Acts::VectorMultiTrajectory>();

    Acts::TrackContainer<Acts::VectorTrackContainer,
                     Acts::VectorMultiTrajectory,
                     std::shared_ptr>
    mutableTracks(trackBackend, trajectoryBackend);

    const std::size_t nAccepted = convertTracks(
      tracccTracks,
      mutableTracks,
      detrayToActsMap,
      m_surfaceMap);

    traccc::edm::track_container<traccc::default_algebra>::const_device tmp(tracccTracks);
    ACTS_INFO("TracccTrackToActsAlg: input has "
          << tmp.tracks.size() << " tracks, "
          << " surface map has " << m_surfaceMap.size() << " surfaces, "
          << " detray map has " << detrayToActsMap.size() << " entries");
    if (tmp.tracks.size() > 0) {
        auto track = tmp.tracks.at(0);
        for (const auto [link_type, state_idx] : track.constituent_links()) {
            auto const& state = tmp.states.at(state_idx);
            auto const& meas = tmp.measurements.at(state.measurement_index());
            std::cerr << "track state surface_link=" << state.smoothed_params().surface_link().value()
                    << " meas surface_link=" << meas.surface_link().value() << "\n";
            break;
        }
    }

    std::cerr << "Looking for surface_link=81346357524212543 in map: "
          << detrayToActsMap.count(81346357524212543ULL) << "\n";

    auto constBackend = std::make_shared<Acts::ConstVectorTrackContainer>(
        std::move(*trackBackend));
    auto constTraj = std::make_shared<Acts::ConstVectorMultiTrajectory>(
        std::move(*trajectoryBackend));

    Acts::TrackContainer<Acts::ConstVectorTrackContainer,
                        Acts::ConstVectorMultiTrajectory,
                        std::shared_ptr>
        constTracks(constBackend, constTraj);

    m_outputActsTracks(ctx, std::move(constTracks));

    ACTS_INFO("TracccTrackToActsAlg: converted "
          << nAccepted << "/ " << tmp.tracks.size() << " tracks");

    return ProcessCode::SUCCESS;
}

}  // namespace ActsExamples