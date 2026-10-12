# Writing scenarios

A scenario is a YAML (or JSON) file with up to six sections. Everything is optional
except a radar and at least the targets you want; anything left out takes the default
in the tables below. Unknown keys are not silently dropped: the loader lists them as
warnings, with a suggestion when the key looks like a typo
(`rain_rate_mmhr` → `rain_rate_mm_hr`).

Coordinates are East-North-Up in metres with the origin anywhere you like
(usually the radar). Azimuths are degrees clockwise from North.
See [Conventions](conventions.md).

A minimal file:

```yaml
scenario:
  name: "Two aircraft"
radar:
  frequency_hz: 3.0e9
  power_watts: 200000
  antenna: {gain_db: 35, beamwidth_az_deg: 1.5, scan_rate_rpm: 12}
targets:
  - name: "Inbound"
    affiliation: hostile
    rcs_m2: 5
    initial_position: {x_m: 20000, y_m: 60000, z_m: 4000}
    velocity: {vx_mps: -100, vy_mps: -250, vz_mps: 0}
```

The fastest way to start a new one is the [scenario editor](interface.md#scenario-editor),
which writes this format.

## scenario

| Key | Default | Meaning |
|---|---|---|
| `name` | `Unnamed Scenario` | shown in the header |
| `description` | | one or two sentences |
| `duration_seconds` | 180 | used by the command-line runner |
| `update_rate_hz` | 30 | engine steps per simulated second |
| `seed` | none | random seed (also accepted under `simulation`) |

## radar

| Key | Default | Meaning |
|---|---|---|
| `name` | `Radar` | |
| `type` | | free text, informational |
| `frequency_hz` | 10e9 | carrier |
| `power_watts` | 100e3 | peak power |
| `prf_hz` | 1000 | pulse repetition frequency |
| `pulse_width_s` | 1e-6 | transmitted pulse length |
| `system_losses_db` | 4 | two-way losses |
| `position` | origin | `{x_m, y_m, z_m}` |
| `height_above_terrain_m` | | if set (and terrain exists), the antenna is placed this high above the ground at `position` |
| `velocity` | 0 | `{vx_mps, vy_mps, vz_mps}` for airborne or ship radars |

`radar.antenna`

| Key | Default | Meaning |
|---|---|---|
| `gain_db` | 30 | |
| `beamwidth_az_deg`, `beamwidth_el_deg` | 2, 3 | 3 dB beamwidths |
| `sidelobe_db` | −30 | one-way sidelobe floor, used for clutter entering through the elevation pattern |
| `polarization_tilt_deg` | 0 | 0 = horizontal, 90 = vertical |
| `scan_type` | `circular` | `circular`, `sector` or `stare` |
| `scan_rate_rpm` / `scan_rate_deg_s` | 6 rpm | rotation rate (circular) |
| `sector_az_min_deg`, `sector_az_max_deg` | −60, 60 | sector limits |
| `scan_speed_deg_s` | rpm × 6 | sector raster speed |
| `stare_azimuth_deg` | 0 | pointing in stare mode |

`radar.receiver`

| Key | Default | Meaning |
|---|---|---|
| `noise_figure_db` | 4 | |
| `bandwidth_hz` | 1/pulse width | noise bandwidth; τ·B > 1 means pulse compression |
| `system_temperature_k` | 290 | |
| `full_scale_dbm` | −20 | hard-limiter level |
| `dynamic_range_db` | | alternative to `full_scale_dbm`: full scale = kTBF + dynamic range |

## targets

A list; each entry:

| Key | Default | Meaning |
|---|---|---|
| `name` | `Target_n` | |
| `type` | `aircraft` | free text (`fighter`, `missile`, `ship`, …) |
| `affiliation` | `unknown` | `friend`, `hostile`, `neutral`, `unknown` (display only) |
| `rcs_m2` | 1 | mean RCS |
| `swerling_model` | 1 | 0–4 |
| `initial_position` | origin | `{x_m, y_m, z_m}`, z = altitude |
| `velocity` | 0 | `{vx_mps, vy_mps, vz_mps}` |
| `acceleration` | 0 | `{ax_mps2, ay_mps2, az_mps2}`: constant acceleration |
| `turn_rate_deg_s` | 0 | constant-rate turn, positive = right |
| `deploy_time_s` | 0 | appears at this time |
| `has_ecm` | false | carries a jammer |
| `ecm_type` | `noise_barrage` | `noise_barrage`, `noise_spot`, `drfm` |
| `ecm_power_watts` | 1000 | jammer ERP |
| `ecm_bandwidth_hz` | 100e6 | noise bandwidth |
| `drfm` | | `{mode: rgpo|vgpo, gain_over_skin_db, capture_dwell_s, pull_rate_mps, max_pull_m, vgpo_rate_hz_per_s, max_doppler_pull_hz, inherent_delay_s}` |

## ecm

Expendables released during the run. With `deployed_by` they start at that target's
position at `deploy_time_s`; otherwise give a position.

```yaml
ecm:
  decoys:
    - name: "MALD"
      deployed_by: "F-16 Alpha"
      deploy_time_s: 30
      velocity: {vx_mps: 300, vy_mps: 30, vz_mps: -5}
      rcs_m2: 5
  chaff_clouds:
    - name: "Chaff 1"
      deployed_by: "F-16 Alpha"
      deploy_time_s: 60
      rcs_m2: 50          # fully bloomed
      bloom_time_s: 1     # rise time constant
      decay_time_s: 120   # dispersal time constant
```

A decoy without its own `velocity` keeps the carrier's velocity; chaff without one
falls at 1 m/s.

## environment

| Key | Default | Meaning |
|---|---|---|
| `temperature_c` | 15 | |
| `pressure_hpa` | 1013.25 | barometric (total) pressure |
| `water_vapor_gpm3` | 7.5 | |
| `rain_rate_mm_hr` | 0 | |
| `rain_height_m` | 3360 | top of the rain layer (ITU-R P.839) |
| `terrain_type` | `rural` | surface class for clutter: `urban`, `suburban`, `rural`, `forest`, `desert`, `mountain`, `sea` |
| `sea_state` | 0 | Douglas 0–6 |
| `ground_surface` | | `{model: gamma|oh1992, gamma_db, relative_permittivity_real, relative_permittivity_loss, rms_height_m}` |
| `clutter.velocity_spread_mps` | land 0.25, sea 1.0 | clutter spectrum width for the MTI improvement factor |
| `clutter.sectors` | | list of `{name, azimuth_start_deg, azimuth_end_deg, range_start_km, range_end_km, terrain_type, sigma0_adjustment_db}`; the first sector containing a cell wins |
| `terrain` | | `{preset: flat|mountainous|ambush}` or a custom relief `{seed, scale_m, max_height_m, octaves, persistence, lacunarity, peaks: [{x_m, y_m, height_m}]}` |

When terrain is given, heights in the file are above mean sea level.

## simulation

| Key | Default | Meaning |
|---|---|---|
| `enable_atmospheric_loss` | true | |
| `enable_clutter` | false | surface and rain clutter |
| `pfa` | 1e-6 | false-alarm probability per cell |
| `pulses_integrated` | 1 | non-coherently integrated pulses per look |
| `range_noise_std_m` | 50 | measurement noise |
| `angle_noise_std_rad` | 0.001 | measurement noise (when monopulse is off) |
| `scan_gated_detection` | true | look only when the beam passes; false = every step |
| `cfar` | off | `{type: CA|GO|SO|OS, reference_cells, guard_cells}` |
| `mti` | off | `{enabled, threshold_mps}` |
| `monopulse` | off | `{enabled}` |
| `eccm` | | `{frequency_agility}` |
| `tracker` | | `{process_noise_mps2: 10, confirm_hits: 3, max_misses: 3}` |
| `seed` | none | random seed for a reproducible run |
