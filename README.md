# BeiGuangAOI

## Requirements and Dependencies
For C++:
- C++ Standard: 2014
- CMake: >= 3.16
- OpenCV: 4.4.0
- Boost: 1.77.0

For Python:
- Python Standard: 3.8
- OpenCV-Python: 4.4.0.46

## Quick Start
For C++, compile the project and run test images:
```bash
$ cd BeiGuangAOI
$ mkdir build && cd build
$ cmake .. && make
$ cd bin && ./test_beiguang_aoi ./images
```

For Python, run the following command:
```bash
$ cd BeiGuangAOI/src/python
$ python3 main.py ../../images
```

All inference results will be saved in "results" folder.
