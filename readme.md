Win10 home,
Python 3.6.5, 
IDE -> jupyter notebook
-------------------------------

資料集：
tai_power_train_1705-1803_utf8 -> 建模的資料集，台電從2017年5月至2018年3月的輸電資料

tai_power_test_1805-1902_utf8 -> 預測的資料集，台電從2018年5月至2019年2月的輸電資料

tai_power_predict_1904_utf8 -> submission 前的資料集，只有日期4/2～4/8，沒有peak_load


預測方法：

模型 -> 2層的LSTM，第一層有128個neurons，dropout為20%，
        第二層有64個neurons，dropout為20%
	最後輸出一個peak_load


訓練-> 利用建模的資料集中每7天的資料來預測下個7天，
       例如用 1/2～1/8預測1/9～1/15，接著再用 1/3～1/9 預測 1/10～1/16，以此類推




