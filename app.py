import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Import your model class here, if necessary

# Function to find the sticker and calculate its diameter in pixels


def find_sticker_diameter_pixels(image):
    # print(image)
    image = (image * 255).astype(np.uint8)
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # Check if any contours were found
    if not cnts:
        raise ValueError("No contours found")

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100:
            continue
        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                 (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 4.0

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # Adjust these values for your sticker's color
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([30, 255, 255])
    # mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # contours, _ = cv2.findContours(
    #     mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # max_diameter = 0
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     perimeter = cv2.arcLength(contour, True)
    #     if perimeter == 0:
    #         continue  # Avoid division by zero
    #     circularity = 4 * np.pi * (area / (perimeter * perimeter))
    #     if area > 100 and circularity > 0.7:
    #         ((x, y), radius) = cv2.minEnclosingCircle(contour)
    #         diameter = radius * 2
    #         max_diameter = max(max_diameter, diameter)

    # print(dA, dB)
    # print(dimA, dimB)
    max_diameter = max(dimA, dimB)
    return max_diameter


app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

# Function to load and preprocess the image


def load_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add any other transforms your model expects
    ])
    return transform(image).unsqueeze(0)

# Function to calculate the scale factor


def calculate_pixel_distance(point1, point2):
    """Calculate the Euclidean distance between two points in pixel space."""
    return np.sqrt(((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2))


def calculate_scale_factor(sticker_diameter_pixels, sticker_real_diameter_inch=4.0):
    return sticker_real_diameter_inch / sticker_diameter_pixels


def get_cow_measurements(keypoints, scale_factor):
    # Use the keypoints to calculate the heart girth and body length in pixels
    # For demonstration, let's assume keypoints 0 and 1 correspond to points needed for body length
    body_length_pixels = calculate_pixel_distance(keypoints[0], keypoints[1])
    # And keypoints 2 and 3 correspond to points needed for heart girth
    heart_girth_pixels = calculate_pixel_distance(keypoints[2], keypoints[3])

    print(body_length_pixels, heart_girth_pixels)

    # Convert pixel measurements to real-world measurements using the scale factor
    body_length_in_inches = body_length_pixels * scale_factor
    heart_girth_in_inches = heart_girth_pixels * scale_factor

    print(body_length_in_inches, heart_girth_in_inches)

    # Apply the Schaeffer formula or other relevant formula to calculate the weight
    weight = schaeffer_formula(body_length_in_inches, heart_girth_in_inches)

    return weight


def schaeffer_formula(body_length, heart_girth):
    # Insert the appropriate formula calculation here
    # This is just a placeholder for the actual formula
    return (heart_girth ** 2 * body_length) / 300


# Endpoint to handle image URL and return keypoints
@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.json
        image_url = json_data['imageUrl']

        # Load and preprocess the image
        image_tensor = load_image(image_url)

        # Convert image tensor to OpenCV format
        image_cv = cv2.cvtColor(
            np.array(image_tensor.squeeze(0).permute(1, 2, 0)), cv2.COLOR_RGB2BGR)

        # Find sticker diameter in pixels
        sticker_diameter_pixels = find_sticker_diameter_pixels(image_cv)

        # Handle case where sticker is not found (sticker_diameter_pixels is zero)
        if sticker_diameter_pixels == 0:
            raise ValueError("Sticker not found or diameter is zero.")

        # Calculate scale factor
        scale_factor = calculate_scale_factor(sticker_diameter_pixels)
        print(scale_factor)

        # Ensure the model and image tensor are on the same device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)

        # Load your model
        model = torch.load('kpts_model.pt', map_location=device)
        model.eval()

        # Predict keypoints
        with torch.no_grad():
            predictions = model(image_tensor)
            # Extract keypoints from predictions, format as needed
            boxes = predictions[0]['boxes'].cpu().numpy().tolist()
            # scores = predictions[0]['scores'].cpu().numpy().tolist()
            # labels = predictions[0]['labels'].cpu().numpy().tolist()
            keypoints = predictions[0]['keypoints'].cpu().numpy().tolist()

        # Calculate weight using keypoints and scale factor
        # Make sure to adjust this to use the correct keypoints for your formula
        estimated_weight = get_cow_measurements(keypoints[0], scale_factor)
        if estimated_weight >= 1000:
            estimated_weight = estimated_weight / 10
        else:
            estimated_weight = estimated_weight
        # Format and return the response
        response = {
            'predictions': {
                'keypoints': keypoints,
                # ... include other prediction details if necessary ...
            },
            'estimated_weight': estimated_weight / 13000,
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
