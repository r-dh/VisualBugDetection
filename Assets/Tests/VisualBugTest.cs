using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Assets.Scripts.BugDetecting;

namespace Tests
{
    public class VisualBugTest
    {
        public float TestDuration = 30; // Wandering duration
        public float Interval = 0.03f; // Check for bugs interval
        public bool Canary = true; // Fail as soon as possible

        private float dt = 0f;

        [UnityTest]
        [Timeout(60000)] //ms (1 min)
        public IEnumerator VisualBugTestEnumerator()
        {
            bool bugs_found = false;

            GameObject world = MonoBehaviour.Instantiate(Resources.Load<GameObject>("Prefabs/Detection/World"));
            yield return new WaitForSeconds(3f);
            Assert.NotNull(world, "World not found.");

            GameObject onnxAgent = MonoBehaviour.Instantiate(Resources.Load<GameObject>("Prefabs/Detection/ONNXAgent"));
            yield return new WaitForSeconds(3f);
            Assert.NotNull(onnxAgent, "ONNX Agent not found.");

            AnalyseFrames af = onnxAgent.GetComponent<AnalyseFrames>();
            Assert.NotNull(af, "AnalyseFrames component not found.");

            // Optionally add (multiple) models here
            // This is a copy of the default initialization
            af.models.Add(new OnnxModel("Assets/AIModels/EfficientNetB0_missing_viking.onnx", "MissingTex", "input_2:0"));

            while (TestDuration > 0)
            {
                if (dt > Interval)
                {  
                    af.AnalyseCurrentFrame(((result) => bugs_found = result || bugs_found), false);
                    dt = 0;
                }
                dt += Time.deltaTime;
                TestDuration -= Time.deltaTime;

                if (Canary) {
                    Assert.IsTrue(bugs_found == false, "Visual bugs were found!"); // Stop at first bug
                }
                yield return null;
            }

            Assert.IsTrue(bugs_found == false, "Visual bugs were found!"); // Assert at end
        }
    }
}
