using NUnit.Framework;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

public class ModelRequest : MonoBehaviour
{
    private string serverUrl = "http://127.0.0.1:5000/predict";

    private float[] inputData = new float[] {  };

    void Start()
    {
        StartCoroutine(SendPredictionRequest(inputData));
    }

    IEnumerator SendPredictionRequest(float[] input)
    {
        string jsonData = "{\"input\": [" + string.Join(",", input) + "]}";

        UnityWebRequest request = new UnityWebRequest(serverUrl,"POST");
        byte[] jsonToSend = new System.Text.UTF8Encoding().GetBytes(jsonData);
        request.uploadHandler = new UploadHandlerRaw(jsonToSend);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string response = request.downloadHandler.text;
            Debug.Log("Response: " + response);
        }
        else {
            Debug.Log("Error: " + request.error);
        }
    }
}
