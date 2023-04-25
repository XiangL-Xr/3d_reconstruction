if  [ ! -d "build/" ];then
  mkdir build
else
  rm -rf build/*
fi
cd build
cmake ..
make -j8
make install
cd ..