from typing import Any, Dict

import numexpr as ne
import numpy as np
from attrs import define, field

from floris.core import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)
from floris.utilities import (
    cosd,
    sind,
    tand,
)

@define
class CosineVelocityDeflection(BaseModel):
    """
    The Gauss deflection model is a blend of the models described in
    :cite:`gdm-bastankhah2016experimental` and :cite:`gdm-King2019Controls` for
    calculating the deflection field in turbine wakes.

    parameter_dictionary (dict): Model-specific parameters.
        Default values are used when a parameter is not included
        in `parameter_dictionary`. Possible key-value pairs include:

            -   **ka** (*float*): Parameter used to determine the linear
                relationship between the turbulence intensity and the
                width of the Gaussian wake shape.
            -   **kb** (*float*): Parameter used to determine the linear
                relationship between the turbulence intensity and the
                width of the Gaussian wake shape.
            -   **alpha** (*float*): Parameter that determines the
                dependence of the downstream boundary between the near
                wake and far wake region on the turbulence intensity.
            -   **beta** (*float*): Parameter that determines the
                dependence of the downstream boundary between the near
                wake and far wake region on the turbine's induction
                factor.
            -   **ad** (*float*): Additional tuning parameter to modify
                the wake deflection with a lateral offset.
                Defaults to 0.
            -   **bd** (*float*): Additional tuning parameter to modify
                the wake deflection with a lateral offset.
                Defaults to 0.
            -   **dm** (*float*): Additional tuning parameter to scale
                the amount of wake deflection. Defaults to 1.0
            -   **use_secondary_steering** (*bool*): Flag to use
                secondary steering on the wake velocity using methods
                developed in [2].
            -   **eps_gain** (*float*): Tuning value for calculating
                the V- and W-component velocities using methods
                developed in [7].
                TODO: Believe this should be removed, need to verify.
                See property on super-class for more details.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
            :keyprefix: gdm-
    """

    alpha: float = field(converter=float, default=0.58)
    beta: float = field(converter=float, default=0.077)
    ka: float = field(converter=float, default=0.38)
    kb: float = field(converter=float, default=0.004)
    eta: float = field(converter=float, default=1.0)


    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
            "freestream_velocity": flow_field.u_initial_sorted,
            "wind_veer": flow_field.wind_veer,
        }
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        rotor_diameter_i: float,
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        freestream_velocity: np.ndarray,
        wind_veer: float,
    ):
        """
        Calculates the deflection field of the wake. See
        :cite:`gdm-bastankhah2016experimental` and :cite:`gdm-King2019Controls`
        for details on the methods used.

        Args:
            x_i (np.array): x-coordinates of turbine i.
            y_i (np.array): y-coordinates of turbine i.
            yaw_i (np.array): Yaw angle of turbine i.
            turbulence_intensity_i (np.array): Turbulence intensity at turbine i.
            ct_i (np.array): Thrust coefficient of turbine i.
            rotor_diameter_i (float): Rotor diameter of turbine i.

        Returns:
            np.array: Deflection field for the wake.
        """
        # ==============================================================



        # Start of the near wake
        xR = x_i

        # Start of the far wake
        x0 = np.ones_like(freestream_velocity)
        x0 *= rotor_diameter_i * cosd(yaw_angle_i) * (1 + np.sqrt(1 - ct_i) )
        x0 /= np.sqrt(2) * (
            4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - np.sqrt(1 - ct_i) )
        )
        x0 += x_i

        # Masks
        # When we have only an inequality, the current turbine may be applied its own
        # wake in cases where numerical precision cause in incorrect comparison. We've
        # applied a small bump to avoid this. "0.1" is arbitrary but it is a small, non
        # zero value.
        near_wake_mask = (x > xR + 0.1) * (x < x0)
        far_wake_mask = (x >= x0)
        upstream_turbine_mask = (x <= xR + 0.1)

        # wake edge at xR
        rwR = rotor_diameter_i*np.sqrt( cosd(yaw_angle_i)*(1+np.sqrt(1-ct_i))/2./np.sqrt(1-ct_i)*0.25 )

        # wake edge at x0
        A = cosd(yaw_angle_i)*(1+np.sqrt(1-ct_i))
        B = 4*(np.pi**2 - 4) - (3*np.pi**2 - 16)*(1 - np.sqrt(1-ct_i))
        rw0 = rotor_diameter_i*np.pi*np.sqrt(A/B)

        # wake expansion rate
        k = 2.*np.pi/np.sqrt(np.pi**2 - 4.)*(self.ka*turbulence_intensity_i + self.kb)
        
        # wake edge
        near_wake_ramp_up = (x - xR) / (x0 - xR)
        near_wake_ramp_down = (x0 - x) / (x0 - xR)

        rw = 0.5 * rotor_diameter_i*upstream_turbine_mask \
            +(rwR*near_wake_ramp_down+rw0*near_wake_ramp_up)*near_wake_mask \
        +(k*(x - x0) + rw0)*far_wake_mask

        # Near wake deflection
        
        tan_theta0 = -0.25*ct_i*sind(yaw_angle_i)/(np.sqrt(1 - ct_i))
        delta_near_wake = self.eta*tan_theta0*(x - xR)*near_wake_mask


        # Far wake deflection
        A=-1.*np.sqrt( ct_i/cosd(yaw_angle_i) )*sind(yaw_angle_i)/8./k
        B1=2*(rw/rotor_diameter_i)-np.sqrt(ct_i*cosd(yaw_angle_i))
        B2=2*(rw/rotor_diameter_i)+np.sqrt(ct_i*cosd(yaw_angle_i))
        C1=2*(rw0/rotor_diameter_i)+np.sqrt(ct_i*cosd(yaw_angle_i))
        C2=2*(rw0/rotor_diameter_i)-np.sqrt(ct_i*cosd(yaw_angle_i))
        delta_far_wake = (self.eta*tan_theta0*(x0 - xR)+rotor_diameter_i*A*np.log(np.clip(B1/B2*C1/C2,1,None)) )*far_wake_mask

        deflection = delta_near_wake + delta_far_wake

        return deflection
    
@define
class CosineVelocityDeficit(BaseModel):

    alpha: float = field(default=0.58)
    beta: float = field(default=0.077)
    ka: float = field(default=0.38)
    kb: float = field(default=0.004)

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:

        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
            "u_initial": flow_field.u_initial_sorted,
            "wind_veer": flow_field.wind_veer
        }
        return kwargs

    # @profile
    def function(
        self,
        x_i: np.ndarray,
        y_i: np.ndarray,
        z_i: np.ndarray,
        axial_induction_i: np.ndarray,
        deflection_field_i: np.ndarray,
        yaw_angle_i: np.ndarray,
        turbulence_intensity_i: np.ndarray,
        ct_i: np.ndarray,
        hub_height_i: float,
        rotor_diameter_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        *,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u_initial: np.ndarray,
        wind_veer: float,
    ) -> None:

        # Compute the bounds of the near and far wake regions and a mask

        # Start of the near wake
        xR = x_i

        # Start of the far wake
        x0 = np.ones_like(u_initial)
        x0 *= rotor_diameter_i * cosd(yaw_angle_i) * (1 + np.sqrt(1 - ct_i) )
        x0 /= np.sqrt(2) * (
            4 * self.alpha * turbulence_intensity_i + 2 * self.beta * (1 - np.sqrt(1 - ct_i) )
        )
        x0 += x_i
        
        # wake edge at xR
        rwR = rotor_diameter_i*np.sqrt( 0.25*cosd(yaw_angle_i)*(1+np.sqrt(1-ct_i))/2./np.sqrt(1-ct_i) )

        # wake edge at x0
        A = cosd(yaw_angle_i)*(1+np.sqrt(1-ct_i))
        B = 4*(np.pi**2 - 4) - (3*np.pi**2 - 16)*(1 - np.sqrt(1-ct_i))
        rw0 = rotor_diameter_i*np.pi*np.sqrt(A/B)

        # Masks
        # When we have only an inequality, the current turbine may be applied its own
        # wake in cases where numerical precision cause in incorrect comparison. We've
        # applied a small bump to avoid this. "0.1" is arbitrary but it is a small, non
        # zero value.
        near_wake_mask = (x > xR + 0.1) * (x < x0)
        far_wake_mask = (x >= x0)
        upstream_turbine_mask = (x <= xR + 0.1)

        # wake expansion rate
        k = 2.*np.pi/np.sqrt(np.pi**2 - 4.)*(self.ka*turbulence_intensity_i + self.kb)

        # wake edge
        near_wake_ramp_up = (x - xR) / (x0 - xR)
        near_wake_ramp_down = (x0 - x) / (x0 - xR)


        rw = 0.5 * rotor_diameter_i*upstream_turbine_mask \
            +(rwR*near_wake_ramp_down+rw0*near_wake_ramp_up)*near_wake_mask \
            +(k*(x - x0) + rw0)*far_wake_mask
         
        #  update wake masks
        near_wake_mask = near_wake_mask*( np.sqrt( (y - y_i - deflection_field_i)**2.+(z - hub_height_i)**2. ) <=rw )
        far_wake_mask =   far_wake_mask*( np.sqrt( (y - y_i - deflection_field_i)**2.+(z - hub_height_i)**2. ) <=rw )

        # Initialize the velocity deficit array
        velocity_deficit = np.zeros_like(u_initial)

        # Compute the velocity deficit in the NEAR&FAR WAKE region

        r, C = rC(
            wind_veer,
            rw,
            y,
            y_i,
            deflection_field_i,
            z,
            hub_height_i,
            ct_i,
            yaw_angle_i,
            rotor_diameter_i,
            near_wake_mask,
            far_wake_mask,
        )
            
        velocity_deficit = cosine_function(C, r, rw, near_wake_mask + far_wake_mask)

        return velocity_deficit


# @profile
def rC(wind_veer, rw, y, y_i, delta, z, HH, Ct, yaw, D,near_wake_mask,far_wake_mask):
    r = np.sqrt( ( (y - y_i - delta) ** 2) + ((z - HH) ** 2) )
    d = np.clip(1 - (3.*np.pi**2-16)*np.pi**2/4./(np.pi**2-4)**2*(Ct * cosd(yaw) / (  rw * rw / (D * D) )), 0.0, 1.0)
    C = 2.*(np.pi**2 - 4.)/(3.*np.pi**2-16)*( 1 - np.sqrt(d) )
    C = ( 1. - np.sqrt(1. - Ct) )*near_wake_mask + C*far_wake_mask
    return r, C

def cosine_function(C, r, rw, wake_mask):
    return C * 0.5 * ( np.cos(np.pi*r/rw) + 1 )*wake_mask

