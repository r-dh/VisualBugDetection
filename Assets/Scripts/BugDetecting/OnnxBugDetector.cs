using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ONNXTest;

namespace Assets.Scripts.BugDetecting
{
    public class OnnxBugDetector
    {
        private string Model;
        public string ModelName;
        private string TensorInputName;
        private float Treshold = 0.1f;

        public static int WIDTH = 224;
        public static int HEIGHT = 224;
        public static int DEPTH = 3;

        private InferenceSession session;

        private static float[] values = new float[WIDTH * HEIGHT * DEPTH];
        private static int[] dimensions = new int[] { 1, WIDTH, HEIGHT, DEPTH };

        public OnnxBugDetector(string model_path, string model_name, string tensor_name = "input_3:0")
        {
            Model = model_path;
            ModelName = model_name;
            TensorInputName = tensor_name;
            session = new InferenceSession(Model);
        }

        public OnnxBugDetector(OnnxModel model)
        {
            Model = model.ModelPath;
            ModelName = model.ModelName;
            TensorInputName = model.TensorInputName;
            Treshold = model.Treshold;
            session = new InferenceSession(Model);
        }

        public async Task<bool> Infer(Tensor<float> input)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(TensorInputName, input)
            };

            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = await Task.Run(() => session.Run(inputs)))
            {
                IEnumerable<float> output = results.First().AsEnumerable<float>();
                IEnumerable<Prediction> pred = output.Select((x, i) => new Prediction { Label = LabelMap.Labels[i], Confidence = x })
                                   .OrderByDescending(x => x.Confidence);
#if UNITY_EDITOR
                string predictions = "";
                foreach (var t in pred)
                {
                    predictions += $"Label: {t.Label}, Confidence: {t.Confidence}\t";
                }
                //Debug.Log(predictions);
#endif
                float bug_certainty = pred.First(p => p.Label == "bug").Confidence;
                float normal_certainty = pred.First(p => p.Label == "normal").Confidence;

                return (bug_certainty - normal_certainty > Treshold); //true is bug

                //return new Tuple<float, float>(normal_certainty, bug_certainty);
            }

        }

        public async Task<bool> Infer(UnityEngine.Color32[] img)
        {
            Tensor<float> input = await Task.Run(() => Color32ToTensor(img));

            return await Infer(input);
        }

        public static Tensor<float> Color32ToTensor(UnityEngine.Color32[] rawimg)
        {            
            UnityEngine.Color32[] img = ArrayPool<UnityEngine.Color32>.Shared.Rent(WIDTH * HEIGHT);
            //Flip upside down over horizontal axis
            for (int i = 0; i < HEIGHT; ++i)
            {
                Array.Copy(rawimg, i * WIDTH, img, (HEIGHT - i - 1) * WIDTH, WIDTH);
            }

            int index = 0;
            for (int i = 0; i < WIDTH * HEIGHT; ++i)
            {
                values[index++] = img[i].r * 1f;
                values[index++] = img[i].g * 1f;
                values[index++] = img[i].b * 1f;
            }

            Tensor<float> input = new DenseTensor<float>(values, dimensions);

            ArrayPool<UnityEngine.Color32>.Shared.Return(img);

            return input;
        }
    }
}
