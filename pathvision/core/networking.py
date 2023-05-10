import requests
import os


def sendResults(results):
    # Define the URL of your Next.js API endpoint
    url = "https://mfum3gsz4cdntt4agk2ordm25q0xsioq.lambda-url.eu-west-2.on.aws/"

    # Define the payload data to be sent in the request body
    payload = {
        'workspace_uuid': os.environ.get("WORKSPACE_UUID"),
        'authToken': os.environ.get("AUTH_TOKEN"),
    }

    # Define the authentication token
    token = os.environ.get('AUTH_TOKEN')

    # Set the request headers
    headers = {
        "Authorization": token,
        "Content-Type": "application/json"
    }

    # Send the POST request with headers
    response = requests.post(url, json=results, headers=headers)

    # Check the response status code
    if response.status_code == 200:
        # Request was successful
        print("POST request was successful")
    else:
        # Request failed
        print("POST request failed:", response.status_code)
