import sys
import time

import cv2
import mss
import numpy as np
import pyautogui

from non_max_suppression import non_max_suppression


def main():
    with mss.mss() as sct:
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        while "Screen capturing":
            x, y = pyautogui.position()
            monitor = {"top": y - 50, "left": x - 50, "width": 96, "height": 96}

            # Get raw pixels from the screen, save it to a Numpy array
            sct_image = sct.grab(monitor)
            np_image = np.array(sct_image)
            image = np.flip(np_image[:, :, :3], 2)
            (H, W) = image.shape[:2]

            blob = cv2.dnn.blobFromImage(
                image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False
            )

            net.setInput(blob)
            (scores, geometry) = net.forward(layerNames)

            rects, confidences = decode_predictions(scores, geometry)

            boxes = non_max_suppression(np.array(rects), 0.5)

            # loop over the bounding boxes
            for (startX, startY, endX, endY) in boxes:
                # in order to obtain a better OCR of the text we can potentially
                # apply a bit of padding surrounding the bounding box -- here we
                # are computing the deltas in both the x and y directions
                padding_ratio = 0.1
                dX = int((endX - startX) * padding_ratio)
                dY = int((endY - startY) * padding_ratio)
                # apply padding to each side of the bounding box, respectively
                startX = max(0, startX - dX)
                startY = max(0, startY - dY)
                endX = min(W, endX + (dX * 2))
                endY = min(H, endY + (dY * 2))
                # extract the actual padded ROI
                roi = image[startY:endY, startX:endX]

                # TODO continue here

                print((startX, startY, endX, endY))
                # draw the bounding box on the image
                cv2.rectangle(np_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # show the output image
            cv2.imshow("Text Detection", np_image)

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break


def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < 0.5:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)


if __name__ == "__main__":
    main()
