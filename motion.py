import numpy as np
import cv2
import csv

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


class MotionEstimator:
    def __init__(self, name, with_gui=False):
        self.orb = cv2.ORB_create(MAX_FEATURES)
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        self.key_points, self.desc = None, None
        self.image = None
        self.mat = None
        self.file_name = name + '.csv'
        self.with_gui = with_gui
        with open(self.file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['name frame homography'])

    @staticmethod
    def mul_2d(a, b):
        m = np.matmul(a[:, :2], b)
        m[:, 2] += a[:, 2]
        return m

    def add(self, image, mask, name, frame_idx):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        key_points, descriptors = self.orb.detectAndCompute(gray, mask)
        if self.key_points is not None:
            matches = self.matcher.match(self.desc, descriptors, None)

            matches.sort(key=lambda x: x.distance, reverse=False)
            count = int(len(matches) * GOOD_MATCH_PERCENT)
            matches = matches[:count]

            # matches_image = cv2.drawMatches(self.image, self.key_points, image, key_points, matches, None)
            # cv2.imshow("matches", matches_image)

            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)
            for i, match in enumerate(matches):
                points1[i, :] = self.key_points[match.queryIdx].pt
                points2[i, :] = key_points[match.trainIdx].pt
            # mat = cv2.estimateRigidTransform(points2, points1, False)
            h, m = cv2.estimateAffinePartial2D(points2, points1)
            # h, m = cv2.findHomography(points2, points1, cv2.RANSAC)
            self.mat = h if self.mat is None else self.mul_2d(h, self.mat)
            with open(self.file_name, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, frame_idx, ' '.join(map(str, self.mat.flatten().tolist()))])
            if self.with_gui:
                height, width, channels = image.shape
                image = np.copy(image)
                # image[~mask.astype(bool)] = 0
                mask[0] = 0
                mask[-1] = 0
                mask[:, 0] = 0
                mask[:, -1] = 0
                mask = cv2.warpAffine(mask, self.mat, (width, height * 2)).astype(bool)
                res = cv2.warpAffine(image, self.mat, (width, height * 2))  # ,flags=cv2.WARP_INVERSE_MAP)
                # mask = cv2.warpPerspective(mask, self.mat, (width, height * 2)).astype(bool)
                # res = cv2.warpPerspective(image, self.mat, (width, height * 2))  # ,flags=cv2.WARP_INVERSE_MAP)
                self.image[mask] = res[mask]
                # cv2.addWeighted(self.image, 0.5, res, 0.5, 0.0)
                cv2.imshow('res', self.image)
        else:
            height, width, channels = image.shape
            self.image = np.zeros((height * 2, width, channels), dtype=image.dtype)
            self.image[:height] = image
            self.image[:height][~mask.astype(bool)] = 0

        self.key_points = key_points
        self.desc = descriptors
