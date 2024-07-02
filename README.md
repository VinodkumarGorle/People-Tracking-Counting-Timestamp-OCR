# ### 407-Person-Detection-ReID-Video-Time-OCR.py 
> Install OpenVINO from the official website.

> https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html


### requirements.txt :

> python3 -m pip install -r requirements.txt



### run-program:

> python3 407-Person-Detection-ReID-Video-Time-OCR.py



### The Lines and Centroids are in the function [ draw_boxes ] in 407-Person-Detection-ReID-Video-Time-OCR.py

### Lines :

> Y-ref is the reference line in the video [Yellow line]
*  Here Y- ref is the reference line for the footroid which the value is 290

> Y-up is the [pink line]which is placed above the reference line 
* Here we can count the people who are go away from the camara (i.e OUT) 
     The value of Y-up is Y-ref+60
     
> Y-down is the [pink line] which is placed below the reference line 
* Here we can count the people who are come to the camara (i.e IN) 
     The value of Y-up is Y-ref-60

	If you want to change the line position you can search the #line in draw_boxes() function 
	there we can provide the all the values of the centroid's , footroid's, headroid's lines,...etc.
	Note:- If you want to change the centroid position then the position of line should also be changed.
	instructions also given in comment section.

### Centroids :

> Red : Which is appearing the person in between above Y-up(pink)line and top of the video

> Blue : Which is appearing the person in between Y-up(pink)line and Y-ref(yellow)line

> Black : Which is appearing the person in between Y-down(pink)line and Y-ref(yellow)line

> Green : Which is appearing the person in between below Y-down(pink)line and bottom of the video

	If you want to change the centroid position you can search the #centroid in draw_boxes() function 
	there we can provide the all the values of the centroid, footroid, headroid,...etc.
	Note:- If you want to change the centroid position then the position of line should also be changed.
	intsructions also given in comment section.

### Inputs :

> Video : You can give the entire path of the video in " video_file " value

> RTSP : You can give the entire link of the RTSP streem in " video_file " value



### Time_Filter.py :

* This file  which is used for detecting the time-stamp in the video/image 
* It is only show the output in the format of text (ie uncomment " print(txts) " in Time_filter.py)



### Delete Duplicates using time_delay :

* If the same Id comes in some time _delay (3 seconds) in IN and OUT then it should be removes from Counts.
* The time_delay  is also be in draw_boxes() function .You can also check by using CTRL+F >> time_delay



### 407-Person-Detection-ReID-Video-Time-OCR.py

> run_person_tracking () :

* This is the main function which is used to run the program.
* source is input of  function 
* flip is used to rotate the video in 180 Degrees.
* The use_popup is true we can see the output in the popup window  if it is false we can't see the output video only results printed in the terminal.

> draw_boxes () :

* In this function we can draw the boxes of each person and also draw the centroids of rectangles and reference lines 
[ which are already provided on the comment section in the program ]



