#============
#Imports and module setup
#============
from dataclasses import dataclass
import math

#============
#Sphere
#============
@dataclass(frozen=True)
class Sphere:
    radius: float
    rotation: float
    gravity: float

    #============
    #Post init
    #============
    def __post_init__(self):
        if not all((math.isfinite(x) for x in (self.radius, self.rotation, self.gravity))):
            raise ValueError('Sphere parameters must be finite')
        if self.radius <= 0 or self.gravity <= 0:
            raise ValueError('Radius and effective gravity must be positive')

    #============
    #Metrics
    #============
    def metrics(self, latitude):
        if not math.isfinite(latitude) or abs(latitude) >= math.pi / 2 - 1e-10:
            raise ValueError('Local equations require latitude strictly away from poles')
        return (self.radius, math.cos(latitude), math.tan(latitude), 2 * self.rotation * math.sin(latitude))

#============
#Localfield
#============
@dataclass(frozen=True)
class LocalField:
    value: float
    longitude: float = 0.0
    latitude: float = 0.0

    #============
    #Post init
    #============
    def __post_init__(self):
        if not all((math.isfinite(x) for x in (self.value, self.longitude, self.latitude))):
            raise ValueError('Field and derivatives must be finite real numbers')

#============
#Balanced height gradient
#============
def balanced_height_gradient(sphere, latitude, velocity):
    if not math.isfinite(velocity):
        raise ValueError('Velocity must be finite')
    a, _, tangent, f = sphere.metrics(latitude)
    return -(a * f * velocity + velocity ** 2 * tangent) / sphere.gravity

#============
#Nonlinear tendency
#============
def nonlinear_tendency(sphere, latitude, u, v, h):
    if h.value <= 0:
        raise ValueError('Layer thickness must be positive')
    a, cosine, tangent, f = sphere.metrics(latitude)
    g = sphere.gravity
    du = -u.value * u.longitude / (a * cosine) - v.value * u.latitude / a + (f + u.value * tangent / a) * v.value - g * h.longitude / (a * cosine)
    dv = -u.value * v.longitude / (a * cosine) - v.value * v.latitude / a - (f + u.value * tangent / a) * u.value - g * h.latitude / a
    dh = -(u.value * h.longitude + h.value * u.longitude) / (a * cosine) - (v.value * h.latitude + h.value * v.latitude - h.value * v.value * tangent) / a
    return (du, dv, dh)

#============
#Linear mode tendency
#============
def linear_mode_tendency(sphere, latitude, mode, base_u, base_h, p, q, r, q_latitude, r_latitude):
    if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0:
        raise ValueError('mode must be a nonnegative integer')
    if base_h.value <= 0 or base_u.longitude != 0 or base_h.longitude != 0:
        raise ValueError('Require positive thickness and axisymmetric base fields')
    expected = balanced_height_gradient(sphere, latitude, base_u.value)
    if not math.isclose(base_h.latitude, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('Base state must satisfy gradient-wind balance')
    if not all((math.isfinite(abs(x)) for x in (p, q, r, q_latitude, r_latitude))):
        raise ValueError('Modal amplitudes and derivatives must be finite')
    a, cosine, tangent, f = sphere.metrics(latitude)
    g, U, H = (sphere.gravity, base_u.value, base_h.value)
    ik = 1j * mode / (a * cosine)
    return (-ik * U * p + (f + (U * tangent - base_u.latitude) / a) * q - ik * g * r, -ik * U * q - (f + 2 * U * tangent / a) * p - g * r_latitude / a, -ik * U * r - ik * H * p - (H * q_latitude + (base_h.latitude - H * tangent) * q) / a)
