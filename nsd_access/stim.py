import numpy as np
import h5py
import ast

from pandas.io.json import json_normalize
from pycocotools.coco import COCO
from skimage import io

from .nsda import NSDAccess

SHOWN_IMG_SIZE = 425


class NSDStimulus(object):
    """ NSDStimulus is a class that encapsulates NSD stimulus information.

    it uses NSDAccess's coco annotation information objects to get image metadata
    """

    def __init__(self, image_index, nsda):
        self.image_index = image_index
        self.nsda = nsda
        # derivative information
        self.nsd_info = self.nsda.stim_descriptions.iloc[self.image_index]

        self.coco_image_id = self.nsd_info['cocoId']
        self.coco_split = self.nsd_info['cocoSplit']

        self.coco_info = self.nsda.coco_annotations[self.coco_split +
                                                    '_instances'].loadImgs([self.coco_image_id])[0]

    def get_coco_annots(self):
        """get_coco_annots gets all the annotations for this image,
        and internalizes them
        """
        for info_type in ['captions', 'person_keypoints', 'instances']:
            coco = self.nsda.coco_annotations[self.coco_split+'_'+info_type]
            coco_annot_IDs = coco.getAnnIds([self.coco_image_id])
            setattr(self, 'coco_annot_'+info_type,
                    coco.loadAnns(coco_annot_IDs))

    def get_coco_category(self):
        """get_coco_category internalizes the coco_categories for this image.
        """
        # self.cat_ids = self.coco_annot_instances.getCatIds()
        coco = self.nsda.coco_annotations[self.coco_split +
                                          '_instances']
        self.cat_ids = coco.getCatIds()
        self.categories = json_normalize(
            coco.loadCats(self.cat_ids))
        self.coco_categories = []
        for cat_id in self.cat_ids:
            this_img_list = coco.getImgIds(catIds=[cat_id])
            if self.coco_image_id in this_img_list:
                this_cat = np.asarray(
                    self.categories[self.categories['id'] == cat_id]['name'])[0]
                self.coco_categories.append(this_cat)

    def get_stim_img(self):
        """get_stim_img gets the stimulus that was shown from NSDAccess's hdf5 file

        Returns
        -------
        numpy.ndarray
            The rgb values of the image
        """
        sf = h5py.File(self.nsda.stimuli_file, 'r')
        self.stim_img = sf.get('imgBrick')[self.image_index]

    def get_coco_img(self, source='coco'):
        """get_coco_img gets the image data from the designated online source

        Parameters
        ----------
        source : str, optional
            Where to get the image data, by default 'coco', options: ['coco', 'flickr']
        """
        self.coco_img = np.asarray(
            io.imread(self.coco_info[source+'_url']), dtype="uint8")

    def _get_annotation_transformation(self):
        """_get_annotation_transformation creates a transformation matrix
        from the cropBox and image size to be applied to annotations
        """
        top, bottom, left, right = ast.literal_eval(self.nsd_info['cropBox'])
        h, w = self.coco_info['height'], self.coco_info['width']
        top_pix, bottom_pix, left_pix, right_pix = \
            (1-top)*h, bottom*h, left*w, (1-right)*w
        scaling_factor = (
            SHOWN_IMG_SIZE/np.array([top_pix-bottom_pix, right_pix-left_pix])).max()

        self.annotation_transform_matrix = np.array([
            [scaling_factor, 0, -left_pix],
            [0, scaling_factor, -bottom_pix],
            [0, 0, 1]
        ])

    def _transform_annotation_keypoints(self, keypoints):
        """_transform_annotation_keypoints transforms the annotation's keypoints from the coco img reference frame into the nsd reference frame.add()

        Parameters
        ----------
        keypoints : list of numpy.ndarray, [2M]
            the points making up this annotation's shape, in coco_img space
        """

        if not hasattr(self, 'annotation_transform_matrix'):
            self._get_annotation_transformation()

        transformed_keypoints = []
        for kp in keypoints:
            poly = np.array(kp).reshape((int(len(kp)/2), 2))
            poly += self.annotation_transform_matrix[:2, -1]
            poly *= self.annotation_transform_matrix[0, 0]
            transformed_keypoints.append(poly)

        return transformed_keypoints

    def transform_annotations(self):
        """transform_annotations NOT FINISHED YET
        """
        if not hasattr(self, 'stim_img'):
            self.get_stim_img()

        for annot in self.coco_annot_instances:
            annot['transformed_segmentation'] = self._transform_annotation_keypoints(
                annot['segmentation'])
