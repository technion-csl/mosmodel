if (!exists("input_file")) input_file='data.csv'
if (!exists("output_file")) output_file='whisker.pdf'

set terminal pdf size 4,4
set output output_file
set termopt enhanced

set datafile separator ","
set xlabel 'table walk overhead'
set ylabel 'relative runtime'
set xrange [0:*]
set yrange [0:*]
set grid x
set grid y
set key inside
set size ratio -1
set xtics rotate by 45 right

set offset 0,0
plot for [file in input_file] file using 2:3:4:5 with xyerrorbars pt 7  ps 0.5 notitle


