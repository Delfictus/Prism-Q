use core::fmt;

macro_rules! id_newtype {
    ($name:ident, $inner:ty) => {
        #[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name($inner);

        impl $name {
            #[must_use]
            pub const fn new(value: $inner) -> Self {
                Self(value)
            }

            #[must_use]
            pub const fn get(self) -> $inner {
                self.0
            }
        }
    };
}

macro_rules! finite_quantity {
    ($name:ident) => {
        #[derive(Debug, Copy, Clone, PartialEq)]
        pub struct $name(f64);

        impl $name {
            pub fn new(value: f64) -> Result<Self, QuantityError> {
                if value.is_finite() {
                    Ok(Self(value))
                } else {
                    Err(QuantityError::NonFinite {
                        quantity: stringify!($name),
                        value,
                    })
                }
            }

            #[must_use]
            pub const fn get(self) -> f64 {
                self.0
            }
        }
    };
    ($name:ident, non_negative) => {
        #[derive(Debug, Copy, Clone, PartialEq)]
        pub struct $name(f64);

        impl $name {
            pub fn new(value: f64) -> Result<Self, QuantityError> {
                if !value.is_finite() {
                    return Err(QuantityError::NonFinite {
                        quantity: stringify!($name),
                        value,
                    });
                }
                if value < 0.0 {
                    return Err(QuantityError::OutOfRange {
                        quantity: stringify!($name),
                        value,
                        min: 0.0,
                        max: f64::INFINITY,
                    });
                }
                Ok(Self(value))
            }

            #[must_use]
            pub const fn get(self) -> f64 {
                self.0
            }
        }
    };
    ($name:ident, positive) => {
        #[derive(Debug, Copy, Clone, PartialEq)]
        pub struct $name(f64);

        impl $name {
            pub fn new(value: f64) -> Result<Self, QuantityError> {
                if !value.is_finite() {
                    return Err(QuantityError::NonFinite {
                        quantity: stringify!($name),
                        value,
                    });
                }
                if value <= 0.0 {
                    return Err(QuantityError::OutOfRange {
                        quantity: stringify!($name),
                        value,
                        min: f64::MIN_POSITIVE,
                        max: f64::INFINITY,
                    });
                }
                Ok(Self(value))
            }

            #[must_use]
            pub const fn get(self) -> f64 {
                self.0
            }
        }
    };
}

id_newtype!(ResidueIdx, u32);
id_newtype!(EdgeIdx, u32);
id_newtype!(ConformerIdx, u16);
id_newtype!(AnalogIdx, u32);
id_newtype!(ScaffoldIdx, u32);
id_newtype!(VoxelIdx, u32);
id_newtype!(HitCount, u32);

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CampaignId(String);

impl CampaignId {
    pub fn new(value: impl Into<String>) -> Result<Self, QuantityError> {
        let value = value.into();
        if value.is_empty() {
            return Err(QuantityError::EmptyIdentifier {
                identifier: "CampaignId",
            });
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

finite_quantity!(CausalCoupling);
finite_quantity!(HysteresisEnthalpy);
finite_quantity!(HydrationVariance, non_negative);
finite_quantity!(SpatialVariance, non_negative);
finite_quantity!(FrustrationPenalty, non_negative);
finite_quantity!(ComplementPenalty, non_negative);
finite_quantity!(PoseUncertainty, non_negative);
finite_quantity!(ScalingConstant, non_negative);
finite_quantity!(AngstromDistance, non_negative);
finite_quantity!(ShearStress, non_negative);
finite_quantity!(ChannelCapacity, positive);
finite_quantity!(HysteresisCapacity, positive);

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct HysteresisPersistence(f64);

impl HysteresisPersistence {
    pub fn new(value: f64) -> Result<Self, QuantityError> {
        if !value.is_finite() {
            return Err(QuantityError::NonFinite {
                quantity: "hysteresis_persist",
                value,
            });
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(QuantityError::OutOfRange {
                quantity: "hysteresis_persist",
                value,
                min: 0.0,
                max: 1.0,
            });
        }
        Ok(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DTSGEdgeMetrics {
    pub te_out: CausalCoupling,
    pub te_in: CausalCoupling,
    pub delta_hc: HysteresisEnthalpy,
    pub sigma_hyd: HydrationVariance,
    pub spatial_var: SpatialVariance,
    pub hysteresis_persist: HysteresisPersistence,
}

impl DTSGEdgeMetrics {
    pub fn new(
        te_out: CausalCoupling,
        te_in: CausalCoupling,
        delta_hc: HysteresisEnthalpy,
        sigma_hyd: HydrationVariance,
        spatial_var: SpatialVariance,
        hysteresis_persist: HysteresisPersistence,
    ) -> Result<Self, QuantityError> {
        Ok(Self {
            te_out,
            te_in,
            delta_hc,
            sigma_hyd,
            spatial_var,
            hysteresis_persist,
        })
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DTSGEdge {
    pub from: ResidueIdx,
    pub to: ResidueIdx,
    pub metrics: DTSGEdgeMetrics,
}

impl DTSGEdge {
    #[must_use]
    pub const fn new(from: ResidueIdx, to: ResidueIdx, metrics: DTSGEdgeMetrics) -> Self {
        Self { from, to, metrics }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantityError {
    EmptyIdentifier {
        identifier: &'static str,
    },
    NonFinite {
        quantity: &'static str,
        value: f64,
    },
    OutOfRange {
        quantity: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },
}

impl fmt::Display for QuantityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier { identifier } => {
                write!(formatter, "{identifier} must not be empty")
            }
            Self::NonFinite { quantity, value } => {
                write!(formatter, "{quantity} must be finite, got {value}")
            }
            Self::OutOfRange {
                quantity,
                value,
                min,
                max,
            } => {
                write!(formatter, "{quantity}={value} outside [{min}, {max}]")
            }
        }
    }
}

impl std::error::Error for QuantityError {}
