<?php

// $array = array(1, 2, 3);
$array = [1, 2, 3]; // 簡易構文
echo $array[0];
echo $array[1];
echo $array[2];

$map = ['A' => 4, 'B' => 5, 'C' => 6];
echo $map['A'];
echo $map['B'];
echo $map['C'];

var_dump($array);
var_dump($map);

// 通常のfor文を使用した配列の繰り返し処理
for ($i = 0; $i < count($array); $i++) {
	echo $array[$i];
}

// foreach文を使用した配列の繰り返し処理
foreach ($array as $value) {
	echo $value;
}

// foreach文を使用した連想配列の繰り返し処理
foreach ($map as $key => $value) {
	// echo $key, ":", $value;
	echo "{$key} : {$value}";
}