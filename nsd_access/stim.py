import numpy as np
from pandas.io.json import json_normalize
from pycocotools.coco import COCO
from .nsda import NSDAccess


class NSDStimulus(object):
    """ NSDStimulus is a class that encapsulates NSD stimulus information.

    it uses NSDAccess's coco annotation information objects to get image metadata
    """

    def __init__(self, image_index, nsda):
        self.image_index = image_index
        self.nsda = nsda
        # derivative information
        self.nsd_info = self.nsda.stim_descriptions.iloc[self.image_index]
        self.coco_image_ID = self.nsd_info['cocoId']
        self.coco_split = self.nsd_info['cocoSplit']

    def get_coco_annots(self):
        """get_coco_annots gets all the annotations for this image,
        and internalizes them
        """
        for info_type in ['captions', 'person_keypoints', 'instances']:
            coco = self.nsda.coco_annotations[self.coco_split+'_'+info_type]
            coco_annot_IDs = coco.getAnnIds([self.coco_image_ID])
            setattr(self, 'coco_annot_'+info_type,
                    coco.loadAnns(coco_annot_IDs))

    def get_coco_category(self):
        """get_coco_category internalizes the coco_categories for this image.
        """
        self.cat_ids = self.coco_annot_instances.getCatIds()
        self.categories = json_normalize(
            self.coco_annot_instances.loadCats(self.cat_ids))
        self.coco_categories = []
        for cat_id in self.cat_ids:
            this_img_list = self.coco_annot_instances.getImgIds(catIds=[
                                                                cat_id])
            if self.coco_image_id in this_img_list:
                this_cat = np.asarray(
                    self.categories[self.categories['id'] == cat_id]['name'])[0]
                self.coco_categories.append(this_cat)
