PSM使用教學
============
本文內的程式架構參考自Chong Dang ( https://github.com/rickydangc/psmatching ) 並再進行調整，

另外新增配對流程需要的其他步驟進入整個架構內，

另外此次本套件使用的估計分數模型為LDA，

可以改用其他sklearn的分類模型。


使用說明
-----------

首先一開始需要指定執行整個配對需要的參數

`psm( k = int , caliper = int )`

k為單一筆實驗組觀察需要配對的控制組觀察值個數。

caliper為限制實驗組與控制組的分數差距，設定為caliper乘上傾向分數的Std。


指定完後需要用指定的物件.fit去套入資料

`fit(data , treatment ,y)`

data為分析所要的資料，未被指定為treatment或y，則會被全部當成控制變數。

treatment為分析所使用的實驗變數。

y為分析所要使用的應變數。


再來介紹套件內可呈現的結果

`.pre_check()`       利用t-test檢定配對前觀察值是否接受處理各控制變數是否顯著。

`.overlap()`         實驗組與控制組傾向分數分佈圖。

`.match_outcome()`   有配對到的實驗組個數。

`.match_quality()`   利用t-test檢定配後前觀察值是否接受處理各控制變數是否顯著。

`.matched_data`      配對後的資料。






