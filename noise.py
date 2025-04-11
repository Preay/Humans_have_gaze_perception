import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=15):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    return noisy

def darken_image(image, factor=0.4):
    return (image * factor).astype(np.uint8)

# Initialize camera
cap = cv2.VideoCapture(0)

print("Press 's' to save clear and noisy image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistent output
    frame = cv2.resize(frame, (640, 480))

    # Create noise variants
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    noisy = add_gaussian_noise(frame)
    dark = darken_image(frame)

    # Stack and show
    stacked = np.hstack((frame, blurred, noisy, dark))
    cv2.imshow("Clear | Blurred | Noisy | Dark", stacked)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("clear.png", frame)
        cv2.imwrite("blurred.png", blurred)
        cv2.imwrite("noisy.png", noisy)
        cv2.imwrite("dark.png", dark)
        print("Images saved as: clear.png, blurred.png, noisy.png, dark.png")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
