# -*- encoding: utf-8 -*-

SCORE_NONE = -1


class Life(object):

      def __init__(self, aGene=None):
            self.gene = aGene
            self.score = SCORE_NONE  # 初始化生命值
