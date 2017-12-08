<?php

// PDOを使用してデータベースに接続
$dsn = "mysql:host=localhost;dbname=test";
$dbuser = "test";
$dbpass = "test";

// 例外処理構文
try { // 例外発生の可能性があるコード
	$db = new PDO($dsn, $dbuser, $dbpass); // データベースへの接続

	// 新規レコードの挿入
	$stmt = $db->prepare("INSERT INTO users VALUES(:id, :name, :pass)");
	$stmt->execute(
		[':id' => 4, ':name' => 'TEST', ':pass' => 'test']);
	displayAll(); //　テーブル・レコードの表示

	$db = null;
} catch (PDOException $e) { // 例外処理
	echo $e->getMessage();
}

// テーブルのレコードをすべて表示する関数
function displayAll() {
	global $db;
	$sql = "SELECT * FROM users";
	foreach ($db->query($sql) as $user) {
		echo "{$user['id']}: {$user['name']}\n";
	}
}