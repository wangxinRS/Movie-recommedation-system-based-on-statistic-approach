# 简介
基于tmdb-5000电影数据集，我们搭建一个电影推荐系统。我们期望系统能够基于电影本身的内容信息，推荐经典高分相对冷门的相似电影。

在这一过程中，我们涉及到数据读取、数据清洗、可视化以及推荐引擎的搭建。

在搭建推荐引擎过程中，我们有两个阶段，召回阶段和过滤阶段。我们从将近5000部电影中基于相似度召回40部电影，然后在过滤阶段对这40部电影打分，选出5部电影作为最终的推荐。

详细介绍：https://editor.csdn.net/md?articleId=107883400
