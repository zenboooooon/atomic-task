using System;
using System.IO;
using TpsTransfer;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace TpsTransfer.Editor
{
    public static class TpsTaskSceneBuilder
    {
        private const string ScenePath = "Assets/Scenes/TpsTask.unity";

        [MenuItem("TPS/Create task scene")]
        public static void CreateTaskScene()
        {
            if (!File.Exists("Assets/StreamingAssets/pilot_layouts.json"))
            {
                throw new FileNotFoundException("pilot layout manifest is missing");
            }
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            floor.name = "Floor";
            floor.transform.position = new Vector3(0f, -0.1f, 0f);
            floor.transform.localScale = new Vector3(12f, 0.2f, 12f);
            SetColor(floor, new Color(0.48f, 0.5f, 0.5f));

            var target = GameObject.CreatePrimitive(PrimitiveType.Cube);
            target.name = "VisibleTarget";
            target.transform.position = new Vector3(0f, 1.6f, 5f);
            target.transform.localScale = new Vector3(1f, 2f, 0.4f);
            SetColor(target, new Color(1f, 0.12f, 0.08f));

            var obstacles = new GameObject[3];
            for (int index = 0; index < obstacles.Length; index++)
            {
                var obstacle = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                obstacle.name = "Obstacle" + index;
                obstacle.transform.position = new Vector3(0f, 0.5f, 20f);
                SetColor(obstacle, new Color(0.25f, 0.25f, 0.28f));
                obstacle.SetActive(false);
                obstacles[index] = obstacle;
            }

            var player = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            player.name = "TaskAgent";
            player.transform.position = new Vector3(0f, 1f, 0f);
            player.transform.localScale = new Vector3(0.7f, 1f, 0.7f);
            SetColor(player, new Color(0.1f, 0.25f, 0.9f));
            var body = player.AddComponent<Rigidbody>();
            body.mass = 1f;
            body.constraints = RigidbodyConstraints.FreezeRotation;
            body.interpolation = RigidbodyInterpolation.Interpolate;

            var yawRig = new GameObject("YawRig").transform;
            yawRig.SetParent(player.transform, false);
            var pitchRig = new GameObject("PitchRig").transform;
            pitchRig.SetParent(yawRig, false);
            var cameraObject = new GameObject("PolicyCamera");
            cameraObject.transform.SetParent(pitchRig, false);
            cameraObject.transform.localPosition = new Vector3(0f, 2.2f, -3.5f);
            var camera = cameraObject.AddComponent<Camera>();
            camera.fieldOfView = 70f;
            camera.nearClipPlane = 0.1f;
            camera.farClipPlane = 50f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.55f, 0.75f, 0.95f);
            cameraObject.AddComponent<AudioListener>();

            var sensor = player.AddComponent<CameraSensorComponent>();
            sensor.Width = 224;
            sensor.Height = 224;
            sensor.Grayscale = false;
            sensor.CompressionType = SensorCompressionType.PNG;
            sensor.ObservationStacks = 1;
            sensor.SensorName = "policy_rgb";
            sensor.Camera = camera;

            var behavior = player.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = "TPS_Task";
            behavior.BehaviorType = BehaviorType.Default;
            behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(4);
            behavior.BrainParameters.VectorObservationSize = 0;

            var agent = player.AddComponent<TpsTaskAgent>();
            var decisions = player.AddComponent<Unity.MLAgents.DecisionRequester>();
            decisions.DecisionPeriod = 5;
            decisions.DecisionStep = 0;
            decisions.TakeActionsBetweenDecisions = false;
            agent.YawRig = yawRig;
            agent.PitchRig = pitchRig;
            agent.PolicyCamera = camera;
            agent.Target = target.transform;
            agent.Obstacles = obstacles;
            agent.LayoutStride = 1;

            var lightObject = new GameObject("DirectionalLight");
            var light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.2f;
            lightObject.transform.rotation = Quaternion.Euler(45f, -30f, 0f);

            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
        }

        [MenuItem("TPS/Build Linux task player")]
        public static void BuildLinuxPlayer()
        {
            CreateTaskScene();
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = "Build/TpsTask.x86_64",
                target = BuildTarget.StandaloneLinux64,
                options = BuildOptions.None
            });
            if (report.summary.result != BuildResult.Succeeded)
            {
                throw new Exception("TPS task player build failed: " + report.summary.result);
            }
        }

        private static void SetColor(GameObject gameObject, Color color)
        {
            if (!AssetDatabase.IsValidFolder("Assets/Materials"))
            {
                AssetDatabase.CreateFolder("Assets", "Materials");
            }
            var path = "Assets/Materials/" + gameObject.name + ".mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                material = new Material(Shader.Find("Standard"));
                AssetDatabase.CreateAsset(material, path);
            }
            material.color = color;
            gameObject.GetComponent<Renderer>().sharedMaterial = material;
        }
    }
}
