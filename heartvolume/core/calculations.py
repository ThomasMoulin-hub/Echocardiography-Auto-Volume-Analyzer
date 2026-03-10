import math

from .models import DopplerResult, VolumeResult


def calculate_volume_simple(length_cm: float, diameter_cm: float) -> float:
    """3/4 ellipsoid approximation: V = pi/6 * D^2 * L."""
    return (math.pi / 6.0) * (diameter_cm ** 2) * length_cm


def calculate_volume_simpson(length_cm: float, d_apex_cm: float, d_pm_cm: float, d_mv_cm: float) -> float:
    """Modified Simpson rule using 3 short-axis diameters."""
    a_apex = math.pi * (d_apex_cm / 2.0) ** 2
    a_pm = math.pi * (d_pm_cm / 2.0) ** 2
    a_mv = math.pi * (d_mv_cm / 2.0) ** 2
    return (length_cm / 3.0) * (a_apex + a_mv + 4.0 * a_pm) / 3.0


def calculate_cardiac_output_doppler(vti_cm: float, d_ao_cm: float, hr_bpm: float) -> float:
    """Returns CO in cm^3/min (numerically equal to mL/min)."""
    area_ao = math.pi * (d_ao_cm / 2.0) ** 2
    return vti_cm * hr_bpm * area_ao


def build_volume_result(edv: float, esv: float, hr_bpm: float | None = None) -> VolumeResult:
    sv = edv - esv
    ef = (sv / edv) * 100.0 if edv > 0 else 0.0
    co = (sv * hr_bpm / 1000.0) if hr_bpm is not None else None
    return VolumeResult(edv=edv, esv=esv, sv=sv, ef=ef, co=co)


def build_doppler_result(d_ao_cm: float, vti_cm: float, hr_bpm: float) -> DopplerResult:
    area_ao = math.pi * (d_ao_cm / 2.0) ** 2
    co_l_min = calculate_cardiac_output_doppler(vti_cm, d_ao_cm, hr_bpm) / 1000.0
    sv_ml = (co_l_min * 1000.0) / hr_bpm
    return DopplerResult(d_ao=d_ao_cm, vti=vti_cm, hr=hr_bpm, area_ao=area_ao, sv=sv_ml, co=co_l_min)

