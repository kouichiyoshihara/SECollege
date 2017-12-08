<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title>Login</title>
	</head>
	<body>
		<h1>Login</h1>
		<form action="login.php" method="post">
			<input name="name" placeholder="Name">
			<input type="password" name="pass" placeholder="Password">
			<input type="submit" value="Login">
		</form>
		<?php 
			require 'db.php';
			if (isset($_POST['name']) && isset($_POST['pass'])) {

				// if ($_POST['name'] === "test" && $_POST['pass'] === "test") {
				if (auth($_POST['name'], $_POST['pass'])) {
					session_start();
					$_SESSION['auth'] = true;
					// リダイレクト(転送)
					header("Location: top.php"); 
				} else {
					echo "<p>Login error.</p>";
				}
			}

		?>
	</body>
</html>