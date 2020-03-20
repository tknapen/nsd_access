from .nsda import NSDAccess
from pycocotools.coco import COCO


class NSDStimulus(object):
    """ NSDStimulus is a class that encapsulates NSD stimulus information.
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
