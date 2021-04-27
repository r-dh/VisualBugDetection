using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using UnityEngine;
using System.Collections;
using UnityEngine.Experimental.Rendering;
using System.Diagnostics;
using Unity.Jobs;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine.Rendering;
using System.Buffers;

namespace Assets.Scripts.BugDetecting
{
    public class AnalyseFrames : MonoBehaviour
    {
        //public bool Record = false;
        //public float Interval = 0.5f;
        public List<OnnxModel> models;

        private Rect rect;
        private RenderTexture renderTexture;
        private Texture2D screenShot;
        private readonly int WIDTH = 224;
        private readonly int HEIGHT = 224;

        private string output_dir;

        List<OnnxBugDetector> bugDetectors = new List<OnnxBugDetector>();

        public void Start()
        {
            Init();
        }

        public void Init()
        {
            if (models.Count == 0)
            {
                models.Add(new OnnxModel("Assets/AIModels/EfficientNetB0_missing_viking.onnx", "MissingTex", "input_2:0"));
            }

            UnityEngine.Debug.Log("Adding " + models[0].ModelName);
            bugDetectors.AddRange(models.Select(model => new OnnxBugDetector(model)));

            string root_dir = Application.persistentDataPath; // Path.GetDirectoryName(Application.dataPath); //Directory.GetCurrentDirectory();
            output_dir = Path.Combine(root_dir, "output");
            Directory.CreateDirectory(output_dir);

            UnityEngine.Debug.Log("Initialized");
        }

        public IEnumerator CaptureFrame(Action<bool> af_callback, bool write_jpg_to_disk)
        {
            if (renderTexture == null)
            {
                rect = new Rect(0, 0, WIDTH, HEIGHT);
                renderTexture = new RenderTexture(WIDTH, HEIGHT, 24);
                screenShot = new Texture2D(WIDTH, HEIGHT, TextureFormat.RGB24, false);
            }

            Camera camera = this.GetComponent<Camera>();
            camera.targetTexture = renderTexture;
            camera.Render();

            RenderTexture.active = renderTexture;
            screenShot.ReadPixels(rect, 0, 0);

            camera.targetTexture = null;
            RenderTexture.active = null;

            _ = EvaluateResult(screenShot, af_callback, write_jpg_to_disk); //fire-and-forget
            yield return new WaitForEndOfFrame(); //Wait for 1 frame between ReadPixels() and GetPixels32() to prevent choking GPU
        }

        public void AnalyseCurrentFrame(Action<bool> callback, bool write_jpg_to_disk)
        {
            StartCoroutine(CaptureFrame(callback, write_jpg_to_disk));
        }

        public async Task EvaluateResult(Texture2D frame, Action<bool> callback, bool write_jpg_to_disk = true)
        {
            Color32[] pixelData32 = frame.GetPixels32();
            byte[] bytes = frame.EncodeToJPG(); //get jpg before inference, otherwise byebye frame

            bool[] results;
            if (bugDetectors.Count > 1) { 
                Tensor<float> data = await Task.Run(() => OnnxBugDetector.Color32ToTensor(pixelData32));
                results = await Task.WhenAll(bugDetectors.Select(bugdetector => bugdetector.Infer(data)));
            } else {
                results = new bool[] { await Task.Run(() => bugDetectors[0].Infer(pixelData32)) };
            }

#if UNITY_EDITOR
            UnityEngine.Debug.Log($"bugs found: { results.Count(r => r == true)}");
#endif
            if (results.Contains(true) && write_jpg_to_disk)
            {
                WriteJPGToDisk(bytes, results);
            }

            callback?.Invoke(results.Contains(true));
        }

        private void WriteJPGToDisk(byte[] jpg, bool[] detector_results)
        {
            string positives = "";
            for (int i = 0; i < detector_results.Length; i++)
            {
                if (detector_results[i])
                {
                    positives += bugDetectors[i].ModelName + "_";
                }
            }

            System.DateTime now = System.DateTime.Now;
            string time_stamp = string.Format("{0}{1}{2}{3}{4}{5}",
                now.Year,
                now.Month,
                now.Day,
                now.Hour,
                now.Minute,
                now.Second);
            string path = string.Format($"{output_dir}/{positives}_{time_stamp}.jpg");

            File.WriteAllBytes(path, jpg);

#if UNITY_EDITOR
            string[] fileArray = Directory.GetFiles($"{path}", "*.jpg");
            UnityEngine.Debug.Log($"Wrote image #{fileArray.Length} to: {path}");     
#endif
        }
    }
}
