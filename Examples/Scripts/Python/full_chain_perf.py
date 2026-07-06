#!/usr/bin/env python3

from pathlib import Path

import acts
import acts.examples
from acts.examples.simulation import (
    MomentumConfig,
    EtaConfig,
    PhiConfig,
    ParticleConfig,
    addParticleGun,
    addPythia8,
    addGenParticleSelection,
    ParticleSelectorConfig,
    addFatras,
    addDigitization,
    addDigiParticleSelection,
)
from acts.examples.reconstruction import (
    SeedingAlgorithm,
    SeedFinderConfigArg,
    SeedFilterConfigArg,
    addSeeding,
    TrackSelectorConfig,
    CkfConfig,
    addCKFTracks,
)
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("outputDir", help="Output directory", type=Path)
parser.add_argument("--ttbar", help="Use ttbar events", action="store_true")
parser.add_argument("--geant4", help="Use geant4 simulation files", type=Path)
args = parser.parse_args()

u = acts.UnitConstants
geoDir = getOpenDataDetectorDirectory()
outputDir = Path.cwd() / args.outputDir
outputDir.mkdir(parents=True, exist_ok=True)

oddMaterialMap = geoDir / "data/odd-material-maps.root"
oddDigiConfig = geoDir / "config/odd-digi-smearing-config.json"
oddSeedingSel = geoDir / "config/odd-seeding-config.json"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

detector = getOpenDataDetector(
    odd_dir=geoDir, materialDecorator=oddMaterialDeco
)
trackingGeometry = detector.trackingGeometry()

field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)


events = 20
runs = 50

if args.ttbar:
    events = 3
    runs = 30

if args.geant4 is not None:
    events = 1
    runs = 30


def create_sequencer():
    s = acts.examples.Sequencer(
        events=events,
        numThreads=1,
        outputDir=str(outputDir),
        trackFpes=False,
        # logLevel=acts.logging.WARNING,
    )

    if args.geant4 is None:
        if not args.ttbar:
            addParticleGun(
                s,
                MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, transverse=True),
                EtaConfig(-3.0, 3.0, uniform=True),
                PhiConfig(0.0, 360.0 * u.degree),
                ParticleConfig(4, acts.PdgParticle.eMuon, randomizeCharge=True),
                vtxGen=acts.examples.GaussianVertexGenerator(
                    mean=acts.Vector4(0, 0, 0, 0),
                    stddev=acts.Vector4(
                        0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 1.0 * u.ns
                    ),
                ),
                multiplicity=50,
                rnd=rnd,
                # outputDirRoot=outputDir,
                # outputDirCsv=outputDir,
            )
        else:
            addPythia8(
                s,
                hardProcess=["Top:qqbar2ttbar=on"],
                npileup=200,
                vtxGen=acts.examples.GaussianVertexGenerator(
                    mean=acts.Vector4(0, 0, 0, 0),
                    stddev=acts.Vector4(
                        0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns
                    ),
                ),
                rnd=rnd,
                # outputDirRoot=outputDir,
                # outputDirCsv=outputDir,
            )

        addGenParticleSelection(
            s,
            ParticleSelectorConfig(
                rho=(0.0, 24 * u.mm),
                absZ=(0.0, 1.0 * u.m),
            ),
        )

        addFatras(
            s,
            trackingGeometry,
            field,
            enableInteractions=True,
            # outputDirRoot=outputDir,
            # outputDirCsv=outputDir,
            rnd=rnd,
        )
    else:
        s.addReader(
            acts.examples.root.RootParticleReader(
                level=acts.logging.WARNING,
                outputParticles="particles_generated_selected",
                filePath=args.geant4 / "particles.root",
            )
        )
        s.addReader(
            acts.examples.root.RootVertexReader(
                level=acts.logging.WARNING,
                outputVertices="vertices_generated",
                filePath=args.geant4 / "vertices.root",
            )
        )
        s.addReader(
            acts.examples.root.RootParticleReader(
                level=acts.logging.WARNING,
                outputParticles="particles_simulated",
                filePath=args.geant4 / "particles_simulation.root",
            )
        )
        s.addReader(
            acts.examples.root.RootSimHitReader(
                level=acts.logging.WARNING,
                outputSimHits="simhits",
                treeName="hits",
                filePath=args.geant4 / "hits.root",
            )
        )

        s.addWhiteboardAlias("particles", "particles_simulated")
        s.addWhiteboardAlias(
            "particles_simulated_selected", "particles_simulated"
        )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=oddDigiConfig,
        # outputDirRoot=outputDir,
        # outputDirCsv=outputDir,
        rnd=rnd,
    )

    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            eta=(-3.0, 3.0),
            pt=(0.9 * u.GeV, None),
            removeNeutral=True,
        ),
    )

    addSeeding(
        s,
        trackingGeometry,
        field,
        seedingAlgorithm=SeedingAlgorithm.GridTriplet,
        geoSelectionConfigFile=oddSeedingSel,
        seedFinderConfigArg=SeedFinderConfigArg(
            r=(33 * u.mm, 200 * u.mm),
            deltaR=(1 * u.mm, 300 * u.mm),
            collisionRegion=(-250 * u.mm, 250 * u.mm),
            z=(-2000 * u.mm, 2000 * u.mm),
            maxSeedsPerSpM=3,
            sigmaScattering=5,
            radLengthPerSeed=0.1,
            minPt=0.8 * u.GeV,
            impactMax=3 * u.mm,
        ),
        seedFilterConfigArg=SeedFilterConfigArg(
            seedConfirmation=True,
        ),
        # outputDirRoot=outputDir,
        # outputDirCsv=outputDir,
    )

    addCKFTracks(
        s,
        trackingGeometry,
        field,
        TrackSelectorConfig(
            pt=(1.0 * u.GeV, None),
            absEta=(None, 3.0),
            loc0=(-4.0 * u.mm, 4.0 * u.mm),
            nMeasurementsMin=7,
            maxHoles=2,
            maxOutliers=2,
            maxHolesAndOutliers=4,
        ),
        CkfConfig(
            chi2CutOffMeasurement=9,
            chi2CutOffOutlier=15,
            numMeasurementsCutOff=1,
            seedDeduplication=True,
            stayOnSeed=True,
        ),
        # writeCovMat=True,
        # outputDirCsv=outputDir,
        # outputDirRoot=outputDir,
        # logLevel=acts.logging.VERBOSE,
    )

    return s


import pandas as pd

s = create_sequencer()

times = []
for i in range(runs):
    print(f"start round {i}")
    s.run()
    d = pd.read_csv(outputDir / "timing.csv")
    t = {}
    if args.geant4 is None:
        t["fatras"] = d[d["identifier"] == "Algorithm:FatrasSimulation"][
            "time_perevent_s"
        ].values[0]
    t["seeding"] = d[d["identifier"] == "Algorithm:GridTripletSeedingAlgorithm"][
        "time_perevent_s"
    ].values[0]
    t["ckf"] = d[d["identifier"] == "Algorithm:TrackFindingAlgorithm"][
        "time_perevent_s"
    ].values[0]
    print(f"finished and got times {t}")
    times.append(t)
    pd.DataFrame(times).to_csv(outputDir / "times.csv", index=False)
