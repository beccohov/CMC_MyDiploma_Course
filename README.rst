|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Исследование масштабирования размера оптимального батча в больших языковых моделях
    :Тип научной работы: НИР
    :Автор: Бессонов Аркадий Сергеевич
    :Научный руководитель: д.ф-м н. Дьяконов Александр Геннадьевич

Abstract
========

В данной статье исследуется связь между оптимальным размером батча при обучении языковых моделей. Нами были проведены эксперименты на широком спектре моделей, варьирующихся от 180M до 4B параметров. Каждая модель обучалась с использованием различных значений размера батча. Количество токенов, применяемых для предварительного обучения модели, зависит только от ее размера и задается величиной $P \times 50$, где $P$ - количество параметров модели. Результаты каждого измерения дали теоретическую оценку оптимального батча для каждой модели, подтверждающую универсальность использованной аппроксимации. Кроме того, была установлена экспериментальная зависимость между размером оптимального теоретического батча и размером модели, а также разработана математическая модель, позволяющая практически оценивать размер батча для больших моделей. Полученные значения приближенно соотносятся с текущими практическими значениями, которые широко принимаются в последнее время.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
