import requests

def sendResults(results, authtoken):
    # Define the URL of your Next.js API endpoint
    url = "https://your-nextjs-api-url.com/api/endpoint"

    # Define the payload data to be sent in the request body

    # Define the authentication token
    token = "your-authentication-token"

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