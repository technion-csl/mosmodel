if (!exists("input_file")) input_file='scatter.csv'
if (!exists("output_file")) output_file='scatter.pdf'
if (!exists("size_ratio")) size_ratio=-1
if (!exists("x_label")) x_label='table walk overhead'
if (!exists("y_label")) y_label='runtime'
if (!exists("legend")) legend=''

set terminal pdf size 4,4
set output output_file
set termopt enhanced

set datafile separator ","
set xlabel x_label
set ylabel y_label
set grid x
set grid y
#set key inside
set key top left
set autoscale
#set size ratio size_ratio
set xtics rotate by 45 right

set offset 0,0
if (!exists("legend")) \
    plot for [file in input_file] file using 2:3 with p pt 7 ps 0.5 notitle ; \
else \
    plot for [i=1:words(input_file)] word(input_file,i) using 2:3 with p pt 7 ps 0.5 title word(legend,i)



