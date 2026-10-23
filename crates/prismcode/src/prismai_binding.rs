//! PrismAI binding schema surface.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PrismAiBinding {
    pub prismcode_layer_id: String,
    pub memory_image_arrow: String,
    pub instruction_stream_arrow: String,
    pub action_validation_required: bool,
}
