import cv2
import os

def take_and_save_picture(capture, label, counter):
    ret, frame = capture.read()
    filename = f"{label}_{counter}.jpg"
    cv2.imwrite(filename, frame)
    with open('labels.txt', 'a') as f:
        f.write(f"{filename}, {label}\n")
    print(f"Image {filename} saved with label {label}")

def main():
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Cannot open camera")
        exit()

    counter = 0
    label = input("Enter initial label: ")

    while True:
        print("Press 'y' to take a picture, 'n' for new label, 'e' to exit: ", end="")
        key = input()

        if key == 'y':
            take_and_save_picture(capture, label, counter)
            counter += 1
        elif key == 'n':
            label = input("Enter new label: ")
            counter = 0
        elif key == 'e':
            break
        else:
            print("Invalid input")

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
