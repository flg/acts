// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/Traccc/TracccChain.hpp"
#include <chrono>

template <typename scalar_t>
using unit = detray::unit<scalar_t>;

namespace ActsExamples {

static traccc::seedfinder_config makeSeedfinderCfg() {
    traccc::seedfinder_config cfg{};
    cfg.collisionRegionMin = -150.f * unit<float>::mm;
    cfg.collisionRegionMax =  150.f * unit<float>::mm;
    // cfg.deltaRMin = 5.f * unit<float>::mm;
    // cfg.deltaRMax = 150.f * unit<float>::mm;
    // cfg.deltaZMax = 600.f * unit<float>::mm;
    // cfg.rMin = 25.f * unit<float>::mm;
    cfg.setup();
    return cfg;
}

static traccc::finding_config makeFindingCfg() {
    traccc::finding_config cfg{};
    cfg.min_track_candidates_per_track = 6;
    cfg.max_track_candidates_per_track = 100;
    cfg.max_num_tracks_per_measurement = 1;
    cfg.min_measurement_voting_fraction = 1.0f;
    // cfg.min_p  *= traccc::unit<float>::GeV;
    // cfg.min_pT *= traccc::unit<float>::GeV;
    cfg.propagation = detray::propagation::config{};
    cfg.max_num_skipping_per_cand = 1;
    return cfg;
}

TracccChain::TracccChain(const std::string& detector_file,
                         const std::string& digitization_file,
                         const std::string& conditions_file,
                         const std::string& material_file,
                         const std::string& grid_file,
                         const std::string& bfield_file)
    : host_mr{},
      cuda_host_mr{},
      device_mr{},
      mr{device_mr, &cuda_host_mr},
      stream{},
      copy{stream.cudaStream()},
      host_copy{},
      host_det_descr{host_mr},
      host_det_cond{host_mr},
      seedfinder_cfg{makeSeedfinderCfg()},
      seedfilter_cfg{},
      spacepoint_grid_cfg{seedfinder_cfg},
      track_params_cfg{},
      finding_cfg{makeFindingCfg()},
      fitting_cfg{},
      resolution_cfg{},
      ca_cuda{
          mr, copy, stream, traccc::clustering_config{},
          traccc::getDefaultLogger("CudaClusteringAlg", traccc::Logging::INFO)},
      ms_cuda{mr, copy, stream,
              traccc::getDefaultLogger("CudaMeasSortingAlg",
                                       traccc::Logging::INFO)},
      sf_cuda{mr, copy, stream,
              traccc::getDefaultLogger("CudaSpFormationAlg",
                                       traccc::Logging::INFO)},
      sa_cuda{
          seedfinder_cfg,
          spacepoint_grid_cfg,
          seedfilter_cfg,
          mr, copy, stream,
          traccc::getDefaultLogger("CudaSeedingAlg", traccc::Logging::INFO)},
      tp_cuda{track_params_cfg, mr, copy, stream,
              traccc::getDefaultLogger("CudaTrackParEstAlg",
                                       traccc::Logging::INFO)},
      finding_cuda{
          finding_cfg,
          mr, copy, stream,
          traccc::getDefaultLogger("CudaFindingAlg", traccc::Logging::INFO)},
      resolution_cuda{resolution_cfg, mr, copy, stream,
                      traccc::getDefaultLogger("CudaAmbiguityResolutionAlg",
                                               traccc::Logging::INFO)},
      fitting_cuda{
          fitting_cfg, mr, copy, stream,
          traccc::getDefaultLogger("CudaFittingAlg", traccc::Logging::INFO)} {

  if (!digitization_file.empty()) {
        traccc::io::read_detector_description(
            host_det_descr, host_det_cond, detector_file,
            digitization_file, conditions_file, traccc::data_format::json);
    }

  traccc::io::read_detector(host_detector, mng_mr, detector_file,
                            material_file, grid_file);

  device_detector =
      traccc::buffer_from_host_detector(host_detector, mng_mr, copy);
  stream.synchronize();

    // std::cerr << "Seedfinder config:"
    //         << "\n rMin=" << seedfinder_cfg.rMin
    //         << "\n rMax=" << seedfinder_cfg.rMax
    //         << "\n zMin=" << seedfinder_cfg.zMin
    //         << "\n zMax=" << seedfinder_cfg.zMax
    //         << "\n deltaRMin=" << seedfinder_cfg.deltaRMin
    //         << "\n deltaRMax=" << seedfinder_cfg.deltaRMax
    //         << "\n deltaZMax=" << seedfinder_cfg.deltaZMax
    //         << "\n collisionRegionMin=" << seedfinder_cfg.collisionRegionMin
    //         << "\n collisionRegionMax=" << seedfinder_cfg.collisionRegionMax
    //         << "\n cotThetaMax=" << seedfinder_cfg.cotThetaMax
    //         << "\n";

    // std::cerr << "Finding config:"
    //       << "\n  min_track_candidates_per_track=" << finding_cfg.min_track_candidates_per_track
    //       << "\n  max_track_candidates_per_track=" << finding_cfg.max_track_candidates_per_track
    //       << "\n  max_num_branches_per_seed=" << finding_cfg.max_num_branches_per_seed
    //       << "\n  max_num_branches_per_surface=" << finding_cfg.max_num_branches_per_surface
    //       << "\n  max_num_skipping_per_cand=" << finding_cfg.max_num_skipping_per_cand
    //       << "\n  max_num_consecutive_skipped=" << finding_cfg.max_num_consecutive_skipped
    //       << "\n  max_num_tracks_per_measurement=" << finding_cfg.max_num_tracks_per_measurement
    //       << "\n  min_measurement_voting_fraction=" << finding_cfg.min_measurement_voting_fraction
    //       << "\n  chi2_max=" << finding_cfg.chi2_max
    //       << "\n  min_p=" << finding_cfg.min_p
    //       << "\n  min_pT=" << finding_cfg.min_pT
    //       << "\n  run_mbf_smoother=" << finding_cfg.run_mbf_smoother
    //       << "\n  duplicate_removal_minimum_length=" << finding_cfg.duplicate_removal_minimum_length
    //       << "\n  min_step_length_for_next_surface=" << finding_cfg.min_step_length_for_next_surface
    //       << "\n  max_step_counts_for_next_surface=" << finding_cfg.max_step_counts_for_next_surface
    //       << "\n";


  traccc::io::read_magnetic_field(host_field, bfield_file);
  device_field = traccc::cuda::make_magnetic_field(host_field);

  device_det_descr = traccc::detector_design_description::buffer{
      [&]() {
        std::vector<unsigned int> sizes(host_det_descr.size());
        for (std::size_t i = 0; i < host_det_descr.size(); ++i) {
          auto this_design = host_det_descr.at(i);
          sizes[i] = std::max(
              static_cast<unsigned int>(this_design.bin_edges_x().size()),
              static_cast<unsigned int>(this_design.bin_edges_y().size()));
        }
        return sizes;
      }(),
      device_mr, &host_mr, vecmem::data::buffer_type::resizable};

  device_det_cond = traccc::detector_conditions_description::buffer{
      static_cast<traccc::detector_conditions_description::buffer::size_type>(
          host_det_cond.size()),
      device_mr};

  copy.setup(device_det_descr)->wait();
  copy(vecmem::get_data(host_det_descr), device_det_descr)->wait();
  copy.setup(device_det_cond)->wait();
  copy(vecmem::get_data(host_det_cond), device_det_cond)->wait();
}

EventResult processEvent(std::shared_ptr<TracccChain> chain,
                         const std::string& data_directory,
                         std::size_t event_id) {
  EventResult result;

  std::cout << "Processing event" << event_id << " on input CSV files." << std::endl;
//   traccc::edm::silicon_cell_collection::host cells{chain->host_mr};
//   static constexpr bool DEDUPLICATE = true;
//   traccc::io::read_cells(
//       cells, event_id, data_directory,
//       traccc::getDefaultLogger("ReadCells", traccc::Logging::INFO),
//       &chain->host_det_cond, traccc::data_format::csv, DEDUPLICATE, false);
//   result.n_cells = cells.size();

    traccc::edm::spacepoint_collection::host sp_host{chain->host_mr};
    traccc::edm::measurement_collection::host meas_host{
            chain->host_mr};

    // Read the hits and measurements from the relevant event files
    traccc::io::read_spacepoints(
        sp_host, meas_host, event_id, data_directory,
        &chain->host_detector,
        &chain->host_det_descr, &chain->host_det_cond, traccc::data_format::csv);

    result.n_spacepoints = sp_host.size();
    result.n_measurements = meas_host.size();

//   traccc::edm::silicon_cell_collection::buffer cells_buf(
//       static_cast<unsigned int>(cells.size()), chain->mr.main);
//   chain->copy.setup(cells_buf)->wait();
//   chain->copy(vecmem::get_data(cells), cells_buf)->wait();

//   auto unsorted_meas = chain->ca_cuda(cells_buf, chain->device_det_descr,
//                                       chain->device_det_cond);
//   traccc::edm::measurement_collection::buffer meas_buf =
//       chain->ms_cuda(unsorted_meas);
//   chain->stream.synchronize();

//   traccc::edm::measurement_collection::host meas_host{chain->host_mr};
//   chain->copy(meas_buf, meas_host)->wait();
//   result.n_measurements = meas_host.size();

//   traccc::edm::spacepoint_collection::buffer sp_buf =
//       chain->sf_cuda(chain->device_detector, meas_buf);
//   chain->stream.synchronize();

//   traccc::edm::spacepoint_collection::host sp_host{chain->host_mr};
//   chain->copy(sp_buf, sp_host)->wait();
//   result.n_spacepoints = sp_host.size();

    // Copy the spacepoint and module data to the device.
    traccc::edm::spacepoint_collection::buffer sp_buf(
        static_cast<unsigned int>(sp_host.size()),
        chain->mr.main);
    chain->copy.setup(sp_buf)->wait();
    chain->copy(vecmem::get_data(sp_host),
                sp_buf)
        ->wait();

    traccc::edm::measurement_collection::buffer
        meas_buf(
            static_cast<unsigned int>(meas_host.size()),
            chain->mr.main);
    chain->copy.setup(meas_buf)->wait();
    chain->copy(vecmem::get_data(meas_host),
                meas_buf)
        ->wait();

  traccc::edm::seed_collection::buffer seeds_buf = chain->sa_cuda(sp_buf);
  chain->stream.synchronize();

  traccc::edm::seed_collection::host seeds_host{chain->host_mr};
  chain->copy(seeds_buf, seeds_host)->wait();
  result.n_seeds = seeds_host.size();

  traccc::bound_track_parameters_collection_types::buffer params_buf =
      chain->tp_cuda(chain->device_field, meas_buf, sp_buf, seeds_buf);
  chain->stream.synchronize();

  traccc::edm::track_container<traccc::default_algebra>::buffer
      track_candidates_buf = chain->finding_cuda(
          chain->device_detector, chain->device_field, meas_buf, params_buf);

  traccc::edm::track_collection<traccc::default_algebra>::host
      track_candidates_host{chain->host_mr};
  chain
      ->copy(track_candidates_buf.tracks, track_candidates_host,
             vecmem::copy::type::device_to_host)
      ->wait();
  result.n_found_tracks = track_candidates_host.size();

//   traccc::edm::track_container<traccc::default_algebra>::buffer resolved_buf =
//       chain->resolution_cuda(track_candidates_buf);

//   traccc::edm::track_collection<traccc::default_algebra>::host resolved_host{
//       chain->host_mr};
//   chain
//       ->copy(resolved_buf.tracks, resolved_host,
//              vecmem::copy::type::device_to_host)
//       ->wait();
//   result.n_resolved_tracks = resolved_host.size();

  traccc::edm::track_container<traccc::default_algebra>::buffer fitted_buf =
      chain->fitting_cuda(chain->device_detector, chain->device_field,
                          track_candidates_buf);
  chain->stream.synchronize();

  traccc::edm::track_collection<traccc::default_algebra>::host fitted_host{
      chain->host_mr};
  chain
      ->copy(fitted_buf.tracks, fitted_host, vecmem::copy::type::device_to_host)
      ->wait();
  result.n_fitted_tracks = fitted_host.size();

  traccc::edm::track_container<traccc::default_algebra>::buffer final_tracks;

  result.measurements.emplace(chain->host_mr);

  const auto measurements_host_tmp =
      chain->copy.to(meas_buf, chain->host_mr, nullptr,
                     vecmem::copy::type::device_to_host);
  chain->host_copy(measurements_host_tmp, result.measurements.value())->wait();

  final_tracks.tracks =
      chain->copy.to(track_candidates_buf.tracks, chain->host_mr, nullptr,
                     vecmem::copy::type::device_to_host);
  final_tracks.states =
      chain->copy.to(track_candidates_buf.states, chain->host_mr, nullptr,
                     vecmem::copy::type::device_to_host);
  final_tracks.measurements =
      vecmem::get_data(result.measurements.value());  // const_view into host

  result.tracks.emplace(std::move(final_tracks));

  result.n_fitted_tracks = chain->host_copy.get_size(
      result.tracks.value().tracks);

  result.detrayToActsMap = chain->detrayToActsMap;
  return result;
}

EventResult processEvent(
    std::shared_ptr<TracccChain> chain,
    traccc::edm::measurement_collection::host&& meas_host,
    traccc::edm::spacepoint_collection::host&& sp_host) {

    EventResult result;
    std::cout << "Processing event on input ACTS measurements." << std::endl;
    result.n_measurements = meas_host.size();
    result.n_spacepoints = sp_host.size();

    // for (std::size_t i = 0; i < meas_host.size(); ++i) {
    //     auto m = meas_host.at(i);
    //     std::cerr << "Traccc meas " << i
    //             << " surface=" << m.surface_link().value()
    //             << " dims=" << m.dimensions()
    //             << " loc0=" << m.local_position()[0]
    //             << " loc1=" << m.local_position()[1]
    //             << " sub0=" << static_cast<int>(m.subspace()[0])
    //             << " sub1=" << static_cast<int>(m.subspace()[1])
    //             << " var0=" << m.local_variance()[0]
    //             << " var1=" << m.local_variance()[1]
    //             << "\n";
    // }

    result.n_spacepoints = sp_host.size();
    result.n_measurements = meas_host.size();

    // Copy to device
    auto const ts_copytodevice = std::chrono::high_resolution_clock::now();
    traccc::edm::spacepoint_collection::buffer sp_buf(
        static_cast<unsigned int>(sp_host.size()), chain->mr.main);
    chain->copy.setup(sp_buf)->wait();
    chain->copy(vecmem::get_data(sp_host), sp_buf)->wait();

    traccc::edm::measurement_collection::buffer meas_buf(
        static_cast<unsigned int>(meas_host.size()), chain->mr.main);
    chain->copy.setup(meas_buf)->wait();
    chain->copy(vecmem::get_data(meas_host), meas_buf)->wait();
    auto const te_copytodevice = std::chrono::high_resolution_clock::now();

    std::cout << "TracccAlg: converted " << sp_host.size()
            << " SPs from " << meas_host.size() << " measurements." << std::endl;

    // for (std::size_t i = 0; i < sp_host.size(); ++i) {
    //     auto sp = sp_host.at(i);
    //     std::cerr << "SP " << i << ": x=" << sp.x() << " y=" << sp.y()
    //             << " z=" << sp.z() << "\n";
    // }

    // Seeding
    auto const ts_seeding = std::chrono::high_resolution_clock::now();
    traccc::edm::seed_collection::buffer seeds_buf = chain->sa_cuda(sp_buf);
    chain->stream.synchronize();
    auto const te_seeding = std::chrono::high_resolution_clock::now();

    traccc::edm::seed_collection::host seeds_host{chain->host_mr};
    chain->copy(seeds_buf, seeds_host)->wait();
    result.n_seeds = seeds_host.size();

    // Track parameter estimation
    auto const ts_trackparams = std::chrono::high_resolution_clock::now();
    traccc::bound_track_parameters_collection_types::buffer params_buf =
        chain->tp_cuda(chain->device_field, meas_buf, sp_buf, seeds_buf);
    chain->stream.synchronize();
    auto const te_trackparams = std::chrono::high_resolution_clock::now();

    // Track finding
    auto const ts_finding = std::chrono::high_resolution_clock::now();
    traccc::edm::track_container<traccc::default_algebra>::buffer
        track_candidates_buf = chain->finding_cuda(
            chain->device_detector, chain->device_field, meas_buf, params_buf);
    auto const te_finding = std::chrono::high_resolution_clock::now();
    result.n_found_tracks = chain->host_copy.get_size(
        track_candidates_buf.tracks);

    // Store results
    auto const ts_copytohost = std::chrono::high_resolution_clock::now();
    result.measurements.emplace(chain->host_mr);
    const auto measurements_host_tmp =
        chain->copy.to(meas_buf, chain->host_mr, nullptr,
                        vecmem::copy::type::device_to_host);
    chain->host_copy(measurements_host_tmp, result.measurements.value())->wait();

    traccc::edm::track_container<traccc::default_algebra>::buffer final_tracks;
    final_tracks.tracks =
        chain->copy.to(track_candidates_buf.tracks, chain->host_mr, nullptr,
                        vecmem::copy::type::device_to_host);
    final_tracks.states =
        chain->copy.to(track_candidates_buf.states, chain->host_mr, nullptr,
                        vecmem::copy::type::device_to_host);
    final_tracks.measurements =
        vecmem::get_data(result.measurements.value());
    auto const te_copytohost = std::chrono::high_resolution_clock::now();

    std::cerr << "Finding: " << chain->host_copy.get_size(final_tracks.tracks)
              << " track candidates from " << seeds_host.size() << " seeds\n";

    result.n_fitted_tracks = chain->host_copy.get_size(final_tracks.tracks);
    result.tracks.emplace(std::move(final_tracks));
    result.detrayToActsMap = chain->detrayToActsMap;

    if (chain->n_events > 0) {
        chain->m_durations["copytodevice"] += (te_copytodevice - ts_copytodevice);
        chain->m_durations["seeding"] += (te_seeding - ts_seeding);
        chain->m_durations["trackparams"] += (te_trackparams - ts_trackparams);
        chain->m_durations["finding"] += (te_finding - ts_finding);
        chain->m_durations["copytohost"] += (te_copytohost - ts_copytohost);
    }
    chain->n_events += 1;

    return result;
}

TracccChain::~TracccChain() {
    // We did not add the duration for the first event:
    n_events -= 1;
    std::cout << "TracccChain timing (average ms per event):"
        << "\n  copytodevice: " << std::chrono::duration<float, std::milli>(
            m_durations["copytodevice"]).count() / n_events
        << "\n  seeding     : " << std::chrono::duration<float, std::milli>(
            m_durations["seeding"]).count() / n_events
        << "\n  trackparams : " << std::chrono::duration<float, std::milli>(
            m_durations["trackparams"]).count() / n_events
        << "\n  finding     : " << std::chrono::duration<float, std::milli>(
            m_durations["finding"]).count() / n_events
        << "\n  copytohost  : " << std::chrono::duration<float, std::milli>(
            m_durations["copytohost"]).count() / n_events
        << std::endl;
}

}  // namespace ActsExamples
