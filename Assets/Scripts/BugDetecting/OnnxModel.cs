using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assets.Scripts.BugDetecting
{
    [Serializable]
    class OnnxModel
    {
        /// <summary>
        /// Path to the onnx file
        /// </summary>
        public string ModelPath = "Assets/Models/EfficientNetB3_viking_converted.onnx";
        /// <summary>
        /// Descriptive name which will be used to save to textures
        /// </summary>
        public string ModelName = "";
        /// <summary>
        /// Minimum difference of category B (bug) over A (normal) to predict B
        /// </summary>
        public float Treshold = 0.1f;
        /// <summary>
        /// Tensor input name as defined in the ONNX file. E.g input_1:0, input_3:0
        /// </summary>
        public string TensorInputName = "input_3:0";

        public OnnxModel(string model_path, string model_name, string tensor_name, float treshold = 0.1f)
        {
            ModelPath = model_path;
            ModelName = model_name;
            TensorInputName = tensor_name;
            Treshold = treshold;
        }
    }
}
