DROP DATABASE IF EXISTS antique;
CREATE DATABASE antique;


use antique;

DROP TABLE IF EXISTS profiles;
DROP TABLE IF EXISTS photos;
DROP TABLE IF EXISTS bids_history;
DROP TABLE IF EXISTS items_comments;
DROP TABLE IF EXISTS items;

DROP TABLE IF EXISTS item_restrictions;
DROP TABLE IF EXISTS item_conditions;
DROP TABLE IF EXISTS sellers;
DROP TABLE IF EXISTS buyers;
DROP TABLE IF EXISTS admins;
DROP TABLE IF EXISTS commissions;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS catalogs;
DROP TABLE IF EXISTS Auctions;







CREATE TABLE users (
  id SERIAL           PRIMARY KEY,
  e_mail VARCHAR(255) UNIQUE   NOT NULL COMMENT 'Е Почта',  
  passwd VARCHAR(255)          NOT NULL COMMENT 'passwd', 
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX (id),
  INDEX (e_mail)
) COMMENT = 'Покупатели Продавцы Админы ...';
-- =====================================================================================================
 


CREATE TABLE commissions (
	id SERIAL PRIMARY KEY,
	rate INT NOT NULL COMMENT 'Комиссия клиента',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
)COMMENT = 'Комиссия клиента';   







CREATE TABLE profiles (
	user_id SERIAL PRIMARY KEY,
    name VARCHAR(255)            NOT NULL COMMENT 'Имя покупателя',
    birthday DATE 				 NOT NULL,
    phone BIGINT 				 NOT NULL,
    adress JSON 				 NOT NULL,
    gender CHAR(1) 				 NOT NULL,
	user_photo_filename VARCHAR(255) DEFAULT 'Android_morda.jpg',
	commission_to_buy_id  BIGINT UNSIGNED NOT NULL DEFAULT '1',
	commission_to_sell_id BIGINT UNSIGNED NOT NULL DEFAULT '1',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
	INDEX (user_id),
	INDEX (phone),
	INDEX (name),
    FOREIGN KEY (user_id) REFERENCES users(id) ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (commission_to_buy_id)  REFERENCES commissions(id) ON UPDATE CASCADE ON DELETE RESTRICT, 
    FOREIGN KEY (commission_to_sell_id) REFERENCES commissions(id) ON UPDATE CASCADE ON DELETE RESTRICT    
)COMMENT = 'Данные участников...'; 
 -- ===================================================================================================== 



CREATE TABLE Auctions (
  	id SERIAL PRIMARY KEY,
  	day_of_auction DATE NOT NULL,
  	name  VARCHAR(255)  NOT NULL UNIQUE COMMENT 'Название аукциона',
  	descr TEXT(255) COMMENT 'Описание',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
)COMMENT = 'Аукционы...';
 -- ===================================================================================================== 



CREATE TABLE catalogs (
  	id SERIAL PRIMARY KEY,
  	name  VARCHAR(255) UNIQUE NOT NULL COMMENT 'Название раздела',
  	descr TEXT(255) COMMENT 'Описание',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) COMMENT = 'Разделы ';
-- ===================================================================================================== 





/*

CREATE TABLE buyers (
	buyer_id BIGINT UNSIGNED NOT NULL,
#	commission_type BIGINT UNSIGNED NOT NULL,
	INDEX (buyer_id),
    FOREIGN KEY (buyer_id) REFERENCES users(id)              ON UPDATE CASCADE ON DELETE RESTRICT ,

    PRIMARY KEY (buyer_id)   
)COMMENT = 'Покупатели';   
-- ===================================================================================================== 



CREATE TABLE sellers (
	seller_id BIGINT UNSIGNED NOT NULL,
#	commission_type BIGINT UNSIGNED NOT NULL,
	INDEX (seller_id),
    FOREIGN KEY (seller_id) REFERENCES users(id)             ON UPDATE CASCADE ON DELETE RESTRICT ,
    
    PRIMARY KEY (seller_id)   
)COMMENT = 'Продавцы';   
-- ===================================================================================================== 

*/

CREATE TABLE admins (
	admin_id BIGINT UNSIGNED NOT NULL,
	permission_level CHAR(1),
    FOREIGN KEY (admin_id) REFERENCES users(id) ON UPDATE CASCADE ON DELETE RESTRICT ,
    PRIMARY KEY (admin_id)   
)COMMENT = 'Администраторы';   
-- ===================================================================================================== 



CREATE TABLE item_conditions (
	id SERIAL PRIMARY KEY,
	rate INT NOT NULL,
	rate_explanation TEXT
)COMMENT = 'Состояние лота';   
-- ===================================================================================================== 



CREATE TABLE item_restrictions (
	id SERIAL PRIMARY KEY,
	-- restriction INT,
	restriction_explanation TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
)COMMENT = 'Необхадимость наличия лицензии на покупку.';   
-- ===================================================================================================== 



CREATE TABLE items (
	id SERIAL PRIMARY KEY,
	is_active ENUM('active', 'sold', 'cancel', 'unsold'),
	seller_id  BIGINT UNSIGNED NOT NULL,
	auction_id BIGINT UNSIGNED NOT NULL,
	catalog_id BIGINT UNSIGNED NOT NULL,	
	estimate DECIMAL (11,2) NOT NULL,
	title VARCHAR(255) 	    NOT NULL,
	desription TEXT         NOT NULL,
	item_condition BIGINT UNSIGNED NOT NULL,
	restriction_id BIGINT UNSIGNED NOT NULL,
	hummer_price DECIMAL (11,2) UNSIGNED  DEFAULT 0,
	winer_id BIGINT UNSIGNED NOT NULL DEFAULT 0,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX (id),
    INDEX (is_active),
    INDEX (seller_id),
    INDEX (auction_id, catalog_id),
    
    INDEX (desription(50)),
    INDEX (winer_id),
    
    FOREIGN KEY (seller_id) REFERENCES users(id)         		  ON UPDATE CASCADE ON DELETE RESTRICT ,
    FOREIGN KEY (winer_id) REFERENCES users(id)                   ON UPDATE CASCADE ON DELETE RESTRICT ,
    FOREIGN KEY (auction_id) REFERENCES Auctions(id)              ON UPDATE CASCADE ON DELETE RESTRICT , 
    FOREIGN KEY (catalog_id) REFERENCES catalogs(id)              ON UPDATE CASCADE ON DELETE RESTRICT ,
    FOREIGN KEY (item_condition) REFERENCES item_conditions(id)   ON UPDATE CASCADE ON DELETE RESTRICT ,
    FOREIGN KEY (restriction_id) REFERENCES item_restrictions(id) ON UPDATE CASCADE ON DELETE RESTRICT     
)COMMENT = 'Информация о лоте';   
 -- =====================================================================================================



CREATE TABLE photos (
  	id SERIAL PRIMARY KEY,
	filename VARCHAR(255) UNIQUE,
	item_id BIGINT UNSIGNED NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,	
	INDEX (item_id),
    FOREIGN KEY (item_id) REFERENCES items(id) ON UPDATE CASCADE ON DELETE RESTRICT   	  
)COMMENT = 'Изображения лота';  
 -- =====================================================================================================



CREATE TABLE bids_history (
  	id SERIAL PRIMARY KEY,
	item_id BIGINT         UNSIGNED NOT NULL,
	bid DECIMAL (11,2)     UNSIGNED NOT NULL,
	who_try_to_buy  BIGINT UNSIGNED NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
	INDEX (item_id),
    FOREIGN KEY (item_id) REFERENCES items(id)               ON UPDATE CASCADE ON DELETE RESTRICT ,  	
    FOREIGN KEY (who_try_to_buy) REFERENCES users(id)  ON UPDATE CASCADE ON DELETE RESTRICT 
)COMMENT = 'История торгов лота';  
 -- =====================================================================================================



CREATE TABLE items_comments (
  	id SERIAL PRIMARY KEY,
	item_id BIGINT UNSIGNED NOT NULL,
	comment TEXT            NOT NULL,
	who_comments  BIGINT UNSIGNED NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,	
	INDEX (item_id),
	INDEX (comment(50)),
    FOREIGN KEY (item_id) REFERENCES items(id)     ON UPDATE CASCADE ON DELETE RESTRICT ,  	
    FOREIGN KEY (who_comments) REFERENCES users(id) ON UPDATE CASCADE ON DELETE RESTRICT 
)COMMENT = 'Комментарии к описанию лота';  















