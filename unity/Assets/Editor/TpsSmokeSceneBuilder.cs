using System;
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
    public static class TpsSmokeSceneBuilder
    {
        private const string ScenePath = "Assets/Scenes/TpsSmoke.unity";

        [MenuItem("TPS/Create smoke scene")]
        public static void CreateSmokeScene()
        {
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

            var player = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            player.name = "SmokeAgent";
            player.transform.position = new Vector3(0f, 1f, 0f);
            player.transform.localScale = Vector3.one;
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
            behavior.BehaviorName = "TPS_Smoke";
            behavior.BehaviorType = BehaviorType.Default;
            behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(4);

            var agent = player.AddComponent<TpsSmokeAgent>();
            agent.YawRig = yawRig;
            agent.PitchRig = pitchRig;

            var lightObject = new GameObject("DirectionalLight");
            var light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.2f;
            lightObject.transform.rotation = Quaternion.Euler(45f, -30f, 0f);

            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            AssetDatabase.SaveAssets();
        }

        [MenuItem("TPS/Build Linux smoke player")]
        public static void BuildLinuxPlayer()
        {
            CreateSmokeScene();
            var options = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = "Build/TpsSmoke.x86_64",
                target = BuildTarget.StandaloneLinux64,
                options = BuildOptions.None
            };
            var report = BuildPipeline.BuildPlayer(options);
            if (report.summary.result != BuildResult.Succeeded)
            {
                throw new Exception("TPS smoke player build failed: " + report.summary.result);
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
