import os
import sys
import cv2
import time
from beiguang_aoi import BeiGuangAOI_Defects_Detector


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 {} <path/to/image_file/or/image_folder>".format(
            sys.argv[0]))
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("The input path, \"{}\" does not exist.".format(
            sys.argv[1]))
        sys.exit(1)

    results_ok_dir = "./results/ok"
    results_ng_dir = "./results/ng"
    os.makedirs(results_ok_dir, exist_ok=True)
    os.makedirs(results_ng_dir, exist_ok=True)

    all_images = []
    if os.path.isdir(sys.argv[1]):
        for img_name in os.listdir(sys.argv[1]):
            img_file = os.path.join(sys.argv[1], img_name)
            if os.path.isfile(img_file):
                all_images.append(img_file)
    else:
        all_images.append(sys.argv[1])

    detector = BeiGuangAOI_Defects_Detector(config_file="../../config.json")
    
    for img_f in all_images:
        start = time.time()
        
        img_data = cv2.imread(img_f)
        img_res, defects = detector.detect(img_data)
        
        end = time.time()

        img_name = os.path.basename(img_f)
        if defects:
            img_save_path = os.path.join(results_ng_dir, img_name)
        else:
            img_save_path = os.path.join(results_ok_dir, img_name)
        cv2.imwrite(img_save_path, img_res)
        
        print("---------------------------------------")
        print("image: {}".format(img_f))
        print("inference time: {:.3f}s".format(end - start))
        print("count of defects: {}".format(len(defects)))
        if defects:
            print("coordinate of defects:")
            for defect in defects:
                cx = defect.cx
                cy = defect.cy
                rw = defect.rw
                rh = defect.rh
                phi = defect.phi
                ra = defect.ra
                print("\t[cx, cy, rw, rh, phi, ra] = "\
                      "[{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}]".format(
                    cx, cy, rw, rh, phi, ra))
        print()


if __name__ == "__main__":
    main()
