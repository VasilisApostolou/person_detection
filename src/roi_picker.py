import cv2
import numpy as np

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"[{x}, {y}],")
        # Draw green dot where you clicked
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("ROI Picker", img)


img_path = "screenshots/cam4_20260807_011403.jpg" 
img = cv2.imread(img_path)

if img is None:
    print("Could not load image. Check the path!")
else:
    img = cv2.resize(img, (640, 360))
    
    cv2.imshow("ROI Picker", img)
    cv2.setMouseCallback("ROI Picker", click_event)
    
    print("Click on the image to select points. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nCopy this array into your app.py:")
    print(np.array(points, dtype=np.int32))