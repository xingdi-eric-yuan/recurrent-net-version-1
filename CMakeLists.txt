cmake_minimum_required(VERSION 3.2)
project( rnn )
find_package( OpenCV REQUIRED )
aux_source_directory( src/. src_list )
add_executable( rnn ${src_list} )
target_link_libraries (rnn ${OpenCV_LIBS})

