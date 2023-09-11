import os
import cv2
import json
import numpy as np

ROI_RATIO_LOW_BOUND = 0.40
ROI_RATIO_UP_BOUND = 0.95


class DefectData:

    def __init__(self, cx, cy, rw, rh, phi, ra):
        self._cx = cx
        self._cy = cy
        self._rw = rw
        self._rh = rh
        self._phi = phi
        self._ra = ra

    @property
    def cx(self):
        return self._cx

    @property
    def cy(self):
        return self._cy

    @property
    def rw(self):
        return self._rw

    @property
    def rh(self):
        return self._rh

    @property
    def phi(self):
        return self._phi

    @property
    def ra(self):
        return self._ra


class BeiGuangAOI_Defects_Detector:

    def __init__(self, config_file=""):
        self.config_file = config_file

        # 参数默认值
        self.gamma_value1 = 4.5
        self.gamma_value2 = 0.25
        self.adapt_const = -35.0
        self.min_defect_area = 6
        self.min_defect_rect_width = 4
        self.min_defect_rect_height = 4

        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                content = json.loads(f.read())
            if "gamma_value1" in content.keys():
                self.gamma_value1 = content["gamma_value1"]
            if "gamma_value2" in content.keys():
                self.gamma_value2 = content["gamma_value2"]
            if "adapt_const" in content:
                self.adapt_const = content["adapt_const"]
            if "min_defect_area" in content:
                self.min_defect_area = content["min_defect_area"]
            if "min_defect_rect_width" in content:
                self.min_defect_rect_width = content["min_defect_rect_width"]
            if "min_defect_rect_height" in content:
                self.min_defect_rect_height = content["min_defect_rect_height"]

        # 计算 Gamma 矫正查询表1：检测 ROI 区域内的非漏光缺陷
        inv_gamma = 1.0 / self.gamma_value1
        self.look_up_table1 = np.array([((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        # 计算 Gamma 矫正查询表2：检测 ROI 区域顶部的漏光缺陷
        inv_gamma = 1.0 / self.gamma_value2
        self.look_up_table2 = np.array([((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

    def _detect_roi(self, img_gray):
        # 自适应二值化：细粒度上分割出 ROI
        thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 21, 0)

        # 去除椒盐噪声：主要为后续计算减少工作量
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 过滤出 ROI 轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_cnts = None
        min_area = 0
        mask_rect = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10**7:
                if min_cnts is None:
                    min_cnts = cnt
                    min_area = area
                else:
                    if area < min_area:
                        min_cnts = cnt
                        min_area = area
        
        # 制作 ROI mask
        if min_cnts is not None:
            x, y, w, h = cv2.boundingRect(min_cnts)
            mask_rect = np.zeros(thresh.shape, dtype=np.uint8)
            cv2.rectangle(mask_rect, (x, y), (x + w, y + h), 255, -1)
        else:
            mask_rect = np.ones(thresh.shape, dtype=np.uint8) * 255
        thresh = cv2.bitwise_and(thresh, mask_rect)

        # 制作 ROI hull
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2048:
                cnts_filtered.append(cnt)
        cnts_stacked = np.vstack([cnt for cnt in cnts_filtered])
        hull = cv2.convexHull(cnts_stacked)

        # 制作 ROI 轮廓的凸包: mask_hull
        mask_hull = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(mask_hull, [hull], -1, 255, -1)

        # 制作 mask_fine: 这个 mask 在巡边上分割很精细，但会丢失粗粒度信息
        mask_fine = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(mask_fine, cnts_filtered, -1, 255, -1)
        mask_fine = cv2.bitwise_and(mask_fine, mask_hull)

        # 找回丢失的粗粒度信息
        # mask_fine 与 mask_hull 的差异，即丢失的粗粒度信息
        mask_diff = cv2.absdiff(mask_fine, mask_hull)
        mask_diff_ori = mask_diff.copy()
        # 找回水平、竖直等边缘信息
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 19))
        mask_diff = cv2.morphologyEx(mask_diff, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))
        mask_diff = cv2.morphologyEx(mask_diff, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_diff = cv2.morphologyEx(mask_diff, cv2.MORPH_OPEN, kernel)
        mask_diff = cv2.absdiff(mask_diff, mask_diff_ori)
        mask_hull[mask_diff != 0] = 0
        
        # 中值滤波，让最终的 mask 平滑
        mask = cv2.medianBlur(mask_hull, 5)

        # 判断当前 ROI mask 是否合理
        nz_count = cv2.countNonZero(mask)
        tot_count = mask.shape[0] * mask.shape[1]
        ratio = nz_count / tot_count
        if ratio > ROI_RATIO_UP_BOUND or ratio < ROI_RATIO_LOW_BOUND:
            mask = None
        
        return mask
    
    def _detect_light_leaking(self, mask_roi, img_roi):
        # 先提取ROI区域的边缘
        edges = cv2.Canny(mask_roi, 100, 200)
        # 水平方向上的边缘
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        edges_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        # 竖直方向上的边缘
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        edges_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)

        # 计算上半区域，在水平方向上的大致位置
        h, w = mask_roi.shape
        edges_top_half = edges_h[:int(h/2), :]
        cnts_top_half, _ = cv2.findContours(edges_top_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_stacked = np.vstack([cnt for cnt in cnts_top_half])
        cnts_stacked = np.squeeze(np.array(cnts_stacked))
        avg_hy = int(np.average(cnts_stacked, axis=0)[1])

        # 计算上半区域，在竖直方向上的大致位置（左侧和右侧）
        # 左侧
        edges_top_left = edges_v[:int(h/2), :int(w/2)]
        cnts_top_left, _ = cv2.findContours(edges_top_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_stacked = np.vstack([cnt for cnt in cnts_top_left])
        cnts_stacked = np.squeeze(np.array(cnts_stacked))
        avg_vx1 = int(np.average(cnts_stacked, axis=0)[0])
        # 右侧
        edges_top_right = edges_v[:int(h/2), int(w/2):]
        cnts_top_right, _ = cv2.findContours(edges_top_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_stacked = np.vstack([cnt for cnt in cnts_top_right])
        cnts_stacked = np.squeeze(np.array(cnts_stacked))
        avg_vx2 = int(np.average(cnts_stacked, axis=0)[0] + w/2)
        
        # 向下偏移
        offset_y1 = 50
        # 向下偏移
        offset_y2 = 400
        roi = img_roi[avg_hy-offset_y1:avg_hy+offset_y2, avg_vx1:avg_vx2]
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        # Gamma 矫正
        cv2.normalize(roi, roi, 0, 255, cv2.NORM_MINMAX)
        roi = cv2.LUT(roi, self.look_up_table2)
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        roi = cv2.medianBlur(roi, 11)

        # 找出漏光区域在宽度方向的范围： xmin 和 xmax
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack([cnt for cnt in contours]))
        mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)
        edges = cv2.Canny(mask, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        nonzeros = cv2.findNonZero(edges)
        xmin = np.squeeze(np.min(nonzeros, axis=0))[0]
        xmax = np.squeeze(np.max(nonzeros, axis=0))[0]
        xmin += avg_vx1
        xmax += avg_vx1

        # 找出漏光区域在高度方向的范围： ymin 和 ymax
        # 左侧向右偏移
        offset_x1 = 800
        # 右侧向左偏移
        offset_x2 = 800
        roi_diff = cv2.bitwise_xor(roi, mask)
        roi_diff_offset = roi_diff[:, offset_x1:w-offset_x2]
        edges = cv2.Canny(roi_diff_offset, 100, 200)
        nonzeros = cv2.findNonZero(edges)
        ymin = np.squeeze(np.min(nonzeros, axis=0))[1]
        ymax = np.squeeze(np.max(nonzeros, axis=0))[1]

        defect = []
        if (ymax - ymin) > 40:
            defect = [xmin, ymin+avg_hy-offset_y1, xmax-xmin, ymax-ymin]
        
        return defect
    
    def detect(self, img_bgr):
        img_out = img_bgr.copy()
        results = []

        # 以下的操作都是对灰度图操作的
        # 只有在绘制缺陷位置时，才使用彩色图（出于无法在灰度图上绘制红色框考虑）
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 获取感兴趣区域
        mask = self._detect_roi(img_gray)
        if mask is None:
            return img_out, results
        img_roi = cv2.bitwise_and(img_gray, mask)

        # NOTE: 仅调试时打开，绘制 ROI
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_out, contours, -1, (255, 0, 0), 1)

        # 消除边界影响
        mask_eroded = mask.copy()
        cv2.drawContours(mask_eroded, contours, -1, 0, 10)
        # NOTE: 仅调试时打开
        cnts, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_out, cnts, -1, (0, 255, 0), 1)

        # ROI内缩
        img_roi = cv2.bitwise_and(img_roi, mask_eroded)
        
        # ------------------- 以下操作为检测 ROI 区域内的缺陷 -------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(~img_roi, cv2.MORPH_GRADIENT, kernel)

        # Gamma 矫正
        cv2.normalize(gradient, gradient, 0, 255, cv2.NORM_MINMAX)
        gradient = cv2.LUT(gradient, self.look_up_table1)

        # 二值化
        thresh = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 19, self.adapt_const)

        # 去除部分噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 消除边界影响
        cv2.drawContours(thresh, cnts, -1, 0, 20)
        
        # 查找缺陷
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rrect = cv2.minAreaRect(cnt)
            rw, rh = rrect[1]
            area = cv2.contourArea(cnt)
            if area <= 10**6 and rw >= self.min_defect_rect_width and \
                rh >= self.min_defect_rect_height:
                cx, cy = rrect[0]
                phi = rrect[2]
                ra = rw * rh
                defect_data = DefectData(cx, cy, rw, rh, phi, ra)
                results.append(defect_data)

        # 检查顶部漏光
        defect = self._detect_light_leaking(mask, img_roi)
        if defect:
            x, y, w, h = defect
            cx = x + w / 2
            cy = y + h / 2
            rw = w
            rh = h
            phi = 0
            ra = w * h

            defect_data = DefectData(cx, cy, rw, rh, phi, ra)
            results.append(defect_data)

            cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 0, 255), 1)

        # 标注缺陷个数
        if len(results) > 0:
            text = "#defects: {}".format(len(results))
            cv2.putText(img_out, text, (200, 200), cv2.FONT_HERSHEY_COMPLEX,
                6, (0, 0, 255), 3)

        return img_out, results
