using OpenCvSharp;
using OpenCvSharp.Dnn;
using UnityEngine;
using UnityEngine.Video;
using System.Collections;
using System.Collections.Generic;
using TMPro;

public class Main: MonoBehaviour
{
    public GameObject textMesh;
    public TextMeshPro mText;
    public MeshRenderer _projectorRenderer;
    private Texture2D _texture;
    private Mat _image;
    private Vec3b[] _imageData;

    private int width;
    private int height;

    public double scale = 1;

    private Net net;
    private double threshold;
    private int stair_width;
    private int stair_height;
    private int stair_x;
    private int stair_y;
    private static readonly string[] Labels = { "Stairs" };

    // Start is called before the first frame update
    void Start()
    {
        GameObject textMesh = GameObject.Find("text");
        mText = textMesh.GetComponent<TMPro.TextMeshPro>();
        mText.SetText("Initializing");

        var texture = ScreenCapture.CaptureScreenshotAsTexture();
        width = texture.width;
        height = texture.height;

        stair_width = 0;
        stair_height = 0;
        stair_x = 0;
        stair_y = 0;
        threshold = 0.25;

        //_texture = new Texture2D((int)(width / scale), (int)(height / scale), TextureFormat.RGBA32, false);
        _texture = new Texture2D((1147), (557), TextureFormat.RGBA32, false);
        _image = new Mat((557), (1147), MatType.CV_8UC3);
        _imageData = new Vec3b[557 * 1147];

        //_projectorRenderer.material.mainTexture = _texture;
        Object.Destroy(texture);

        var m_Path = Application.dataPath;
        var cfg = m_Path + "/YOLO/yolo-voc.cfg";
        var weights = m_Path + "/YOLO/yolo-voc_best.weights";
        net = CvDnn.ReadNetFromDarknet(cfg, weights);
    }

    // Update is called once per frame
    void Update()
    {
        StartCoroutine(RecordFrame());
    }

    IEnumerator RecordFrame()
    {
        yield return new WaitForEndOfFrame();
        var texture = ScreenCapture.CaptureScreenshotAsTexture();
        // do something with texture
        var matRes = TextureToMat(texture);

        // Assign the Vec3b array to Mat
        _image.SetArray(0, 0, matRes);

        var blob = CvDnn.BlobFromImage(_image, 1 / 255.0, new Size(1147, 575), new Scalar(), true, false);
        net.SetInput(blob, "data");
        var prob = net.Forward();

        predict(prob);

        //MatToTexture();

        // cleanup
        Object.Destroy(texture);
    }

    private Vec3b[] TextureToMat(Texture2D texture)
    {
        var width = texture.width;
        var height = texture.height;

        // Color32 array : r, g, b, a
        Color32[] c = texture.GetPixels32();

        // Convert Color32 object to Vec3b object
        // Vec3b is the representation of pixel for Mat
        for (int i = 0; i < height; i++)
        {
            for (var j = 0; j < width; j++)
            {
                var col = c[j + i * width];
                var vec3 = new Vec3b
                {
                    Item0 = col.b,
                    Item1 = col.g,
                    Item2 = col.r
                };
                // set pixel to an array
                _imageData[j + i * width] = vec3;
            }
        }

        return _imageData;
    }

    private void predict(Mat prob)
    {
        const int prefix = 5;   //skip 0~4
        int w = 1147;
        int h = 557;

        for (int i = 0; i < prob.Rows; i++)
        {
            var confidence = prob.At<float>(i, 4);
            if (confidence > threshold)
            {
                //get classes probability
                Cv2.MinMaxLoc(prob.Row[i].ColRange(prefix, prob.Cols), out _, out Point max);
                var classes = max.X;
                var probability = prob.At<float>(i, classes + prefix);

                if (probability > threshold) //more accuracy
                {
                    var color = Scalar.RandomColor();
                    //get center and width/height
                    var centerX = prob.At<float>(i, 0) * w;
                    var centerY = prob.At<float>(i, 1) * h;
                    var width = prob.At<float>(i, 2) * w;
                    var height = prob.At<float>(i, 3) * h;
                    //label formating
                    var label = $"{Labels[classes]} {probability * 100:0.00}%";
                    print($"confidence {confidence * 100:0.00}% {label}, detected position x:{centerX}, y:{centerY}, width:{width} height:{height}");

                    mText.text = string.Format("Stairs Detected, Confidence {0}", confidence);
                    stair_width = (int)width;
                    stair_height = (int)height;
                    stair_x = (centerX - width / 2) < 0 ? 0 : (int)(centerX - width / 2);
                    stair_y = (int)(centerY - height / 2);

                    UnityEngine.Rect rect = ScreenRect(stair_x, stair_y, stair_width, stair_height);
                    GUI.Label(rect, "TEST");
                }
                else
                {
                    mText.text = "No stairs detected.";
                    stair_width = 0;
                    stair_height = 0;
                    stair_x = 0;
                    stair_y = 0;
                }
            }
        }
    }

    Vector3 ScreenToWorld(float x, float y)
    {
        Camera camera = Camera.current;
        Vector3 s = camera.WorldToScreenPoint(transform.position);
        return camera.ScreenToWorldPoint(new Vector3(x, camera.pixelHeight - y, s.z));
    }

    UnityEngine.Rect ScreenRect(int x, int y, int w, int h)
    {
        Vector3 tl = ScreenToWorld(x, y);
        Vector3 br = ScreenToWorld(x + w, y + h);
        return new UnityEngine.Rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    }

    private void MatToTexture()
    {
        int width = 1147;
        int height = 557;

        // Create Color32 array that can be assigned to Texture2D directly
        Color32[] c = new Color32[width * height];

        // Parallel for loop
        for (int i = 0; i < height; i++)
        {
            for (var j = 0; j < width; j++)
            {
                Vec3b vec = _imageData[j + i * width];
                var color32 = new Color32
                {
                    r = vec[0],
                    g = vec[1],
                    b = vec[2],
                    a = 0
                };
                c[j + i * width] = color32;
            }
        }

        _texture.SetPixels32(c);

        // Update the texture, OpenGL manner
        _texture.Apply();
    }

}
