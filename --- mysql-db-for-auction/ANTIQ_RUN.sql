
use antique;




-- ==========================================================================================================================
-- Вывод всей информации по конкретному лоту.


SET @what_item := 107;

PREPARE info_about_item FROM 
'SELECT  
items.is_active,
Auctions.name as Auction_name,
Auctions.day_of_auction as Auction_date, 
catalogs.name as Catalog_name,
items.estimate ,
items.title,
items.desription,
item_conditions.rate as lot_condition,
item_conditions.rate_explanation as lot_explanation,
item_restrictions.restriction_explanation as lot_restriction

FROM items JOIN Auctions ON items.auction_id=Auctions.id 
JOIN catalogs ON items.catalog_id=catalogs.id 
JOIN item_conditions ON items.item_condition=item_conditions.id
JOIN item_restrictions ON items.restriction_id=item_restrictions.id
WHERE items.id = ?';

EXECUTE info_about_item USING @what_item;



PREPARE info_about_item_comments FROM 
'SELECT items_comments.item_id, comment, name, items_comments.created_at
	FROM items_comments JOIN profiles ON items_comments.who_comments=profiles.user_id 
	WHERE items_comments.item_id =? ORDER BY items_comments.created_at DESC';

EXECUTE info_about_item_comments USING @what_item;



PREPARE info_about_item_bits FROM 
'SELECT item_id, bid, 
	(SELECT name FROM profiles WHERE profiles.user_id=bids_history.who_try_to_buy) as who,
	created_at
	FROM bids_history WHERE item_id =? ORDER BY created_at DESC';
SET @what_item := 107;
EXECUTE info_about_item_bits USING @what_item;




PREPARE item_photos FROM 
	'SELECT item_id, filename FROM photos WHERE item_id=?';

EXECUTE item_photos USING @what_item;	


-- ==========================================================================================================================
-- Сбор и вывод информации по всем лотам конкретного аукциона.


SET @last_auc_id:= (SELECT id FROM Auctions ORDER BY Auctions.day_of_auction DESC LIMIT 1);

SELECT @last_auc_id;


DROP TABLE IF EXISTS temp;
CREATE TEMPORARY TABLE temp 
(
SELECT 
items.is_active,
Auctions.id as Auction_id,
Auctions.name as Auction_name,
Auctions.day_of_auction as Auction_date, 
catalogs.id as Catalog_id,
catalogs.name as Catalog_name,
items.estimate ,
items.title,
items.desription,
item_conditions.rate as lot_condition,
item_conditions.rate_explanation as lot_explanation,
item_restrictions.restriction_explanation as lot_restriction

FROM items JOIN Auctions ON items.auction_id=Auctions.id 
JOIN catalogs ON items.catalog_id=catalogs.id 
JOIN item_conditions ON items.item_condition=item_conditions.id
JOIN item_restrictions ON items.restriction_id=item_restrictions.id
ORDER BY Auction_date DESC, Catalog_name
);


SELECT * FROM temp WHERE Auction_id = @last_auc_id;  #WHERE is_active = 'active' 
-- ==========================================================================================================================
-- Сбор и вывод информации по всем лотам конкретного каталога конкретного аукциона.

SET @catalog_id:=9;
SELECT * FROM temp WHERE Auction_id = @last_auc_id AND Catalog_id =@catalog_id ;  #WHERE is_active = 'active' 

-- ==========================================================================================================================

-- Статистика.
-- Сколько лотов в каких каталогах.
-- Сколько лотов в каких аукционах.

SELECT COUNT(*) as cnt, Catalog_name FROM temp GROUP BY Catalog_name ORDER BY cnt DESC;
SELECT COUNT(*)as cnt, Auction_name FROM temp GROUP BY Auction_name ORDER BY cnt DESC;

-- ==========================================================================================================================
-- Добавление нового пользователя.



DROP PROCEDURE IF EXISTS `sp_add_user`;
DELIMITER //
CREATE PROCEDURE `sp_add_user`(
	email varchar(100), 
	passwd VARCHAR(255), 	
	name VARCHAR(255),
    birthday DATE,
    phone BIGINT,
    adress JSON,
    gender CHAR(1),
	user_photo_filename VARCHAR(255),
	OUT new_user_insert varchar(200))
	
BEGIN
    DECLARE `_rollback` BOOL DEFAULT 0;
   	DECLARE code varchar(30);
   	DECLARE error_string varchar(150);
    DECLARE last_user_id BIGINT;

	DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
	   begin
	    	SET `_rollback` = 1;
			GET stacked DIAGNOSTICS CONDITION 1
	          code = RETURNED_SQLSTATE, error_string = MESSAGE_TEXT;
	    	set new_user_insert := concat('Error occured. Code: ', code, '. Text: ', error_string);
	   end;
		        
	START TRANSACTION;
			INSERT INTO users (e_mail, passwd)
			       VALUES (email, passwd);
			SET @last_user_id := last_insert_id(); 
			INSERT INTO profiles (user_id, name, birthday, phone, adress, gender, user_photo_filename)
			       VALUES (@last_user_id, name, birthday, phone, adress, gender, user_photo_filename); 		
	IF `_rollback` THEN
		ROLLBACK;
	ELSE
		set new_user_insert := 'ok';
			COMMIT;
	END IF;
END//
DELIMITER ;

DELIMITER //
DROP TRIGGER IF EXISTS check_profile_update //
CREATE TRIGGER check_profile_update BEFORE UPDATE ON profiles
FOR EACH ROW
BEGIN

SET NEW.name = COALESCE(NEW.name, OLD.name);
SET NEW.birthday = COALESCE(NEW.birthday, OLD.birthday);
SET NEW.phone = COALESCE(NEW.phone, OLD.phone);
SET NEW.adress = COALESCE(NEW.adress, OLD.adress);
SET NEW.gender = COALESCE(NEW.gender, OLD.gender);
SET NEW.user_photo_filename = COALESCE(NEW.user_photo_filename, OLD.user_photo_filename);



END//

DELIMITER ;


-- вызываем процедуру
call sp_add_user('email@uuu.com', 'passwd', 'name', '1971-05-26 11:47:07', 777777, '{"city": "NY", "street": "Lenina", "h.": "5"}',
	'M', 'user_photo.jpg', @new_user_insert);
SELECT @new_user_insert;


call sp_add_user('email1@uuu.com', 'passwd', 'name3', '1971-05-26 11:47:07', 777777, '{"city": "NY", "street": "Lenina", "h.": "5"}',
	'M', null, @new_user_insert);
SELECT @new_user_insert;
SELECT * FROM users JOIN profiles ON users.id=profiles.user_id;


-- ==========================================================================================================================
-- Добавление нового лота.



DROP PROCEDURE IF EXISTS `sp_add_item`;
DELIMITER //
CREATE PROCEDURE `sp_add_item`(	
	is_active ENUM('active', 'sold', 'cancel', 'unsold'),
	seller_id_num  BIGINT ,
	auction_id BIGINT ,
	catalog_id BIGINT ,	
	estimate DECIMAL (11,2) ,
	title VARCHAR(255) ,
	desription TEXT    ,
	item_condition BIGINT ,
	restriction_id BIGINT ,
		
	OUT new_lot_insert_result varchar(200))
	
BEGIN
   	DECLARE code varchar(30);
   	DECLARE error_string varchar(150);
    DECLARE if_seller_id_exist INT;
    DECLARE if_user_id_exist   INT;
   
	DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
	   begin	   
			GET stacked DIAGNOSTICS CONDITION 1
	          code = RETURNED_SQLSTATE, error_string = MESSAGE_TEXT;
	    	set new_lot_insert_result := concat('Error occured. Code: ', code, '. Text: ', error_string);
	   end;
	  
	SET if_user_id_exist = (SELECT id FROM users WHERE id=seller_id_num);	  
	INSERT INTO items (id, is_active, seller_id, auction_id, catalog_id, estimate, title, 
   				desription, item_condition, restriction_id, winer_id, created_at, updated_at)  				
	      VALUES (DEFAULT, is_active, seller_id_num, auction_id, catalog_id, estimate, title, 
   				desription, item_condition, restriction_id, '1', DEFAULT, DEFAULT);	
	SET new_lot_insert_result := 'ok';	
END//
DELIMITER ;


DELIMITER //
DROP TRIGGER IF EXISTS check_item_update //
CREATE TRIGGER check_item_update BEFORE UPDATE ON items
FOR EACH ROW
BEGIN

SET NEW.is_active = COALESCE(NEW.is_active, OLD.is_active);
SET NEW.seller_id = COALESCE(NEW.seller_id, OLD.seller_id);
SET NEW.auction_id = COALESCE(NEW.auction_id, OLD.auction_id);
SET NEW.is_active = COALESCE(NEW.is_active, OLD.is_active);
SET NEW.catalog_id = COALESCE(NEW.catalog_id, OLD.catalog_id);
SET NEW.estimate = COALESCE(NEW.estimate, OLD.estimate);
SET NEW.title = COALESCE(NEW.title, OLD.title);
SET NEW.desription = COALESCE(NEW.desription, OLD.desription);
SET NEW.item_condition = COALESCE(NEW.item_condition, OLD.item_condition);
SET NEW.restriction_id = COALESCE(NEW.restriction_id, OLD.restriction_id);
SET NEW.hummer_price = COALESCE(NEW.hummer_price, OLD.hummer_price);
SET NEW.winer_id = COALESCE(NEW.winer_id, OLD.winer_id);


END//

DELIMITER ;

-- вызываем процедуру
call sp_add_item('active','100','1','1','333.00','An Italian shiavona',
'Queen. \'It proves nothing of tumbling down stairs! How brave they\'ll all think me at all.\' \'In that case,\' said the Gryphon. \'The reason is,\' said the Hatter: \'I\'m on the whole party look so grave.',
'1', '3', @new_lot_insert_result);
SELECT @new_lot_insert_result;


SELECT * FROM items  JOIN profiles ON items.seller_id=profiles.user_id;



-- ==========================================================================================================================
-- Добавление нового bid-a.


DROP FUNCTION IF EXISTS `get_last_bid`;
DELIMITER //
CREATE FUNCTION `get_last_bid`(check_item_id INT)
RETURNS FLOAT READS SQL DATA
BEGIN
	DECLARE last_bid FLOAT;
	SET last_bid = (
					SELECT  bid FROM bids_history WHERE item_id =check_item_id ORDER BY created_at DESC LIMIT 1
				   );
	RETURN last_bid;
END//
DELIMITER ;


DROP FUNCTION IF EXISTS `get_last_bid_user_id`;
DELIMITER //
CREATE FUNCTION `get_last_bid_user_id`(check_item_id INT)
RETURNS FLOAT READS SQL DATA
BEGIN
	DECLARE last_bid_user_id int;
	SET last_bid_user_id = (
					SELECT  who_try_to_buy FROM bids_history WHERE item_id =check_item_id ORDER BY created_at DESC LIMIT 1
				   );
	RETURN last_bid_user_id;
END//
DELIMITER ;

#select get_last_bid_user_id(107);
#select get_last_bid(107);



DROP PROCEDURE IF EXISTS `sp_new_bid`;
DELIMITER //
CREATE PROCEDURE `sp_new_bid`(	
	item_id BIGINT     ,
	bid DECIMAL (11,2) ,
	who_try_to_buy  BIGINT, 	
	OUT insert_bid_result varchar(200))
	
BEGIN
   	DECLARE code varchar(30);
   	DECLARE error_string varchar(150);
    DECLARE last_bid FLOAT;
   	DECLARE active varchar(150);   
    
	DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
	   begin	  
			GET stacked DIAGNOSTICS CONDITION 1
	          code = RETURNED_SQLSTATE, error_string = MESSAGE_TEXT;
	    	SET insert_bid_result := concat('Error occured. Code: ', code, '. Text: ', error_string);
	   end;
	SET active := (SELECT is_active FROM items WHERE items.id=107);
	IF active = 'active' THEN  
	
		SET last_bid = get_last_bid(item_id);
		IF last_bid >= bid OR bid <= 0 THEN   
	   		SET insert_bid_result := 'invalid value of bid';   
		ELSE
		   INSERT INTO bids_history (id, item_id, bid, who_try_to_buy, created_at)				
			      VALUES       (DEFAULT, item_id, bid, who_try_to_buy, DEFAULT);
			SET insert_bid_result := 'ok';     
		END IF;		
	ELSE
		SET insert_bid_result := 'trading closed';   
	END IF;
	
	
	END//
DELIMITER 



-- вызываем процедуру
call sp_new_bid('101','10000.00','1', @insert_bid_result);
SELECT @insert_bid_result;

call sp_new_bid('107','10000.00','1', @insert_bid_result);
SELECT @insert_bid_result;

#SELECT * FROM bids_history  JOIN profiles ON bids_history.who_try_to_buy=profiles.user_id;



-- ==========================================================================================================================
-- Итоги торгов для конкретного лота.


#select get_last_bid_user_id(148);
#select get_last_bid(148);


SET @lot_id :=107;
SELECT * FROM items WHERE items.id =107;


DROP PROCEDURE IF EXISTS `sp_update_item_results`;
DELIMITER //
CREATE PROCEDURE `sp_update_item_results`(	
	lot_id 			BIGINT ,	
	OUT trading_lot_result varchar(200))
	
BEGIN
    DECLARE `_rollback` BOOL DEFAULT 0;
   	DECLARE code varchar(30);
   	DECLARE error_string varchar(150);
    DECLARE state ENUM('active', 'sold', 'cancel', 'unsold');
    DECLARE winer  					varchar(150) ;
	DECLARE hummer_price_result 	FLOAT  ;
	DECLARE	active ENUM('active', 'sold', 'cancel', 'unsold');
  
   
	DECLARE CONTINUE HANDLER FOR SQLEXCEPTION
	   begin
	    
			GET stacked DIAGNOSTICS CONDITION 1
	          code = RETURNED_SQLSTATE, error_string = MESSAGE_TEXT;
	    	set trading_lot_result := concat('Error occured. Code: ', code, '. Text: ', error_string);
	   end;
		      
	  
	set hummer_price_result = get_last_bid(lot_id);
	set winer               = get_last_bid_user_id(lot_id);	
	set state = (SELECT is_active FROM items WHERE id=lot_id);
	

	IF (winer IS NULL) AND (state = 'active') THEN
		UPDATE items SET is_active = 'unsold', winer_id='1', hummer_price = '0' where items.id = lot_id;
	END IF;
	
	IF (winer IS NOT NULL) AND (state = 'active') THEN
		UPDATE items SET is_active = 'sold', winer_id=winer, hummer_price = hummer_price_result where items.id = lot_id;
	END IF;
	set trading_lot_result := 'ok';

END//
DELIMITER ;



-- вызываем процедуру
call sp_update_item_results(107, @trading_lot_result);
SELECT @trading_lot_result;
SELECT * FROM items WHERE items.id =107;


-- ==========================================================================================================================
-- Итоги торгов для всех лотов всех аукционов.

-- Исходное состояние
SELECT *  FROM items;


DROP TABLE IF EXISTS temp;
CREATE TEMPORARY TABLE temp (SELECT * FROM items);


DROP PROCEDURE IF EXISTS sp_update_all_lots;
DELIMITER //
CREATE PROCEDURE sp_update_all_lots ()
BEGIN

DECLARE is_end INT DEFAULT 0;

DECLARE	id BIGINT;
DECLARE	is_active ENUM('active', 'sold', 'cancel', 'unsold');
DECLARE	seller_id  BIGINT;
DECLARE	auction_id BIGINT;
DECLARE	catalog_id BIGINT;
DECLARE	estimate DECIMAL (11,2);
DECLARE	title VARCHAR(255);
DECLARE	desription TEXT;
DECLARE	item_condition BIGINT;
DECLARE	restriction_id BIGINT;
DECLARE	hummer_price DECIMAL (11,2);
DECLARE	winer_id BIGINT;
DECLARE	created_at DATETIME;
DECLARE updated_at DATETIME;

DECLARE curcat CURSOR FOR SELECT * FROM temp;
DECLARE CONTINUE HANDLER FOR NOT FOUND SET is_end = 1;

OPEN curcat;
	cycle : LOOP
		FETCH curcat INTO id, is_active, seller_id, auction_id, catalog_id, estimate, title, desription, item_condition,
		restriction_id, hummer_price, winer_id, created_at, updated_at;
		IF is_end THEN LEAVE cycle;
		END IF;
		call sp_update_item_results(id, @udate_result);
		
	END LOOP cycle;
CLOSE curcat;
END//
DELIMITER ;

call sp_update_all_lots();

-- Конечное состояние
SELECT *  FROM items;

-- ==========================================================================================================================

-- результаты по всем проданным лотам и их покупателям
DROP TABLE IF EXISTS temp_buyers;
CREATE TEMPORARY TABLE temp_buyers (
SELECT 
items.id as item_id,
winer_id as winer_id,
name as winer_name,
phone as winer_phone,
(SELECT users.e_mail FROM users  WHERE winer_id = users.id ) as winer_mail,
(SELECT Auctions.name FROM Auctions  WHERE Auctions.id = items.auction_id ) as auction_name,
(SELECT catalogs.name FROM catalogs  WHERE catalogs.id = items.catalog_id ) as catalog_name,
title,
hummer_price,
(SELECT commissions.rate FROM commissions JOIN profiles 
	ON profiles.commission_to_buy_id=commissions.id WHERE winer_id = profiles.user_id) as winer_commssion
FROM items JOIN profiles ON winer_id = profiles.user_id WHERE items.is_active = 'sold'
);

#DELETE FROM temp WHERE temp.winer_id=1;
#DELETE FROM temp WHERE temp.seller_id=1;
#SELECT * FROM temp_buyers;

-- результаты по всем проданным лотам и их продавцам
DROP TABLE IF EXISTS temp_sellers;
CREATE TEMPORARY TABLE temp_sellers (
SELECT 
items.id as item_id,
seller_id ,
name as seller_name,
phone as seller_phone,
(SELECT users.e_mail FROM users  WHERE seller_id = users.id ) as seller_mail,
(SELECT Auctions.name FROM Auctions  WHERE Auctions.id = items.auction_id ) as auction_name,
(SELECT catalogs.name FROM catalogs  WHERE catalogs.id = items.catalog_id ) as catalog_name,
title,
hummer_price,
(SELECT commissions.rate FROM commissions JOIN profiles 
	ON profiles.commission_to_sell_id=commissions.id WHERE winer_id = profiles.user_id) as seller_commssion
FROM items JOIN profiles ON seller_id = profiles.user_id WHERE items.is_active = 'sold'
);

#SELECT * FROM temp_sellers;







-- ==========================================================================================================================
-- Список участников купившых лоты.


DROP TABLE IF EXISTS buyers;
CREATE TEMPORARY TABLE buyers 
(
SELECT DISTINCT winer_id FROM items  WHERE items.is_active = 'sold' 
and winer_id >1  #winer_1d = 1 это дефолтный а не реальный пользователь.
ORDER BY winer_id
);
#SELECT * FROM buyers ;









-- ==========================================================================================================================
-- Список участников продавших лоты.
DROP TABLE IF EXISTS sellers;
CREATE TEMPORARY TABLE sellers 
(
SELECT DISTINCT seller_id FROM items  WHERE items.is_active = 'sold' 
and seller_id >1  #seller_id = 1 это дефолтный а не реальный пользователь.
ORDER BY seller_id
);
#SELECT * FROM sellers ;






-- ==========================================================================================================================
-- Список счетов к оплате для участников купивших лоты.

DROP TABLE IF EXISTS invoices_for_buyers;
CREATE TABLE invoices_for_buyers (
	buyer_id BIGINT UNSIGNED NOT NULL,
	to_pay   BIGINT UNSIGNED NOT NULL,
	explntn  Text,
	is_paid  ENUM('YES', 'NOT'),
	INDEX (buyer_id),
    PRIMARY KEY (buyer_id)   
)COMMENT = 'Покупатели'; 


-- ==========================================================================================================================
-- Список счетов к оплате для участников продавших лоты.

DROP TABLE IF EXISTS invoices_for_sellers;
CREATE TABLE invoices_for_sellers (
	seller_id BIGINT UNSIGNED NOT NULL,
	to_pay    BIGINT UNSIGNED NOT NULL,
	explntn   Text,
	is_paid   ENUM('YES', 'NOT'),
	INDEX (seller_id),
    PRIMARY KEY (seller_id)   
)COMMENT = 'Продавцы'; 









-- ==========================================================================================================================
-- Список  лотов конкретного покупателя.


DROP TABLE IF EXISTS temp_buyer;
CREATE TEMPORARY TABLE temp_buyer (
SELECT 
items.id as item_id,
winer_id as winer_id,
name as winer_name,
phone as winer_phone,
(SELECT users.e_mail FROM users  WHERE winer_id = users.id ) as winer_mail,
(SELECT Auctions.name FROM Auctions  WHERE Auctions.id = items.auction_id ) as auction_name,
(SELECT catalogs.name FROM catalogs  WHERE catalogs.id = items.catalog_id ) as catalog_name,
title,
hummer_price,
(SELECT commissions.rate FROM commissions JOIN profiles 
	ON profiles.commission_to_buy_id=commissions.id WHERE winer_id = profiles.user_id) as winer_commssion
FROM items JOIN profiles ON winer_id = profiles.user_id WHERE items.winer_id = 39 AND items.is_active = 'sold'
);


-- ==========================================================================================================================
-- Сбор покупок конкретного покупателя в инвойс.


DROP PROCEDURE IF EXISTS sp_buyer_invoice;
DELIMITER //
CREATE PROCEDURE sp_buyer_invoice (
    	
	OUT invoice_result varchar(200))
BEGIN

DECLARE is_end INT DEFAULT 0;

DECLARE item_id			BIGINT;
DECLARE winer_id		BIGINT;
DECLARE winer_name		VARCHAR(255) ;
DECLARE winer_phone		BIGINT;
DECLARE winer_mail		VARCHAR(255);
DECLARE auction_name	VARCHAR(255);
DECLARE catalog_name	VARCHAR(255);
DECLARE title			VARCHAR(255);
DECLARE hummer_price	DECIMAL (11,2);
DECLARE winer_commssion	INT;
DECLARE content	text;
DECLARE cnt int;
DECLARE lots_summ DECIMAL (11,2);
DECLARE is_paid  ENUM('YES', 'NOT');

DECLARE curcat CURSOR FOR SELECT * FROM temp_buyer;
DECLARE CONTINUE HANDLER FOR NOT FOUND SET is_end = 1;
SET cnt = 0;
SET lots_summ = 0.00;
SET is_paid = 'NOT';
SET content := '{';
OPEN curcat;
		cycle : LOOP
			FETCH curcat INTO item_id, winer_id, winer_name, winer_phone, winer_mail, auction_name, catalog_name,
			title, hummer_price, winer_commssion;
			#SET invoice_result := invoice_result + 1;
			#SET invoice_result := hummer_price;
			IF is_end THEN LEAVE cycle;
			END IF;
			IF cnt = 0 THEN 
				SET content := concat(content, ' "N',cnt,'" : "', item_id, '", "title',cnt,'" : "', title, '" , "hummer_price',cnt,':"  : "', hummer_price);
				SET cnt = cnt + 1;
				SET lots_summ = lots_summ + hummer_price;
			ELSE
				SET content := concat(content, '",  "N',cnt,'" : "', item_id, '",  "title',cnt,'" : "', title, ' ", "hummer_price',cnt,':"  : "', hummer_price);
				SET cnt = cnt + 1;
				SET lots_summ = lots_summ + hummer_price;
			END IF;
			#call sp_update_item_results(id, @udate_result);
			
		END LOOP cycle;
		SET content := concat(content, '"  }');
		SET lots_summ :=lots_summ*(100+winer_commssion)/100;
		INSERT INTO invoices_for_buyers VALUES (winer_id, lots_summ, content, is_paid);
		SET invoice_result := 'ok';
CLOSE curcat;
END//
DELIMITER ;


call sp_buyer_invoice( @invoice_result);
SELECT @invoice_result;
SELECT * FROM invoices_for_buyers ;





















-- ==========================================================================================================================
-- Сбор всех инвойсов всех покупателей.
-- Список счетов к оплате для участников купивших лоты.
DROP TABLE IF EXISTS invoices_for_buyers;
CREATE TABLE invoices_for_buyers (
	buyer_id BIGINT UNSIGNED NOT NULL,
	to_pay   BIGINT UNSIGNED NOT NULL,
	explntn  JSON,
	is_paid  ENUM('YES', 'NOT'),
	INDEX (buyer_id),
	INDEX (is_paid),	
    PRIMARY KEY (buyer_id)   
)COMMENT = 'Покупатели'; 

/*
SELECT * FROM buyers ;
TRUNCATE buyers;
INSERT INTO buyers SELECT DISTINCT winer_id FROM items  WHERE items.is_active = 'sold' ORDER BY winer_id;
SELECT * FROM buyers ;
*/


DROP PROCEDURE IF EXISTS sp_buyers_invoices;
DELIMITER //
CREATE PROCEDURE sp_buyers_invoices (
    	
	OUT invoice_result varchar(200))
BEGIN


DECLARE is_end INT DEFAULT 0;
DECLARE id_of_buyer BIGINT ;

DECLARE curcat CURSOR FOR SELECT * FROM buyers;
DECLARE CONTINUE HANDLER FOR NOT FOUND SET is_end = 1;

OPEN curcat;
		cycle : LOOP
			FETCH curcat INTO id_of_buyer;

			IF is_end THEN LEAVE cycle;
			END IF;
			
		
			DROP TABLE IF EXISTS temp_buyer;
			CREATE TEMPORARY TABLE temp_buyer (
			SELECT 
			items.id as item_id,
			winer_id as winer_id,
			name as winer_name,
			phone as winer_phone,
			(SELECT users.e_mail FROM users  WHERE winer_id = users.id ) as winer_mail,
			(SELECT Auctions.name FROM Auctions  WHERE Auctions.id = items.auction_id ) as auction_name,
			(SELECT catalogs.name FROM catalogs  WHERE catalogs.id = items.catalog_id ) as catalog_name,
			title,
			hummer_price,
			(SELECT commissions.rate FROM commissions JOIN profiles 
				ON profiles.commission_to_buy_id=commissions.id WHERE winer_id = profiles.user_id) as winer_commssion
			FROM items JOIN profiles ON winer_id = profiles.user_id WHERE items.winer_id = id_of_buyer AND items.is_active = 'sold'
			);		
		
		
			
			call sp_buyer_invoice( @invoice_result);
			
		END LOOP cycle;

		SET invoice_result := 'ok';
CLOSE curcat;
END//
DELIMITER ;



call sp_buyers_invoices( @invoice_result);
SELECT @invoice_result;
SELECT * FROM invoices_for_buyers ;





-- ==========================================================================================================================
-- Список  лотов конкретного продавца.


DROP TABLE IF EXISTS temp_seller;
CREATE TEMPORARY TABLE temp_seller (
SELECT 
items.id as item_id,
seller_id as seller_id,
name as seller_name,
phone as seller_phone,
(SELECT users.e_mail FROM users  WHERE seller_id = users.id ) as seller_mail,
(SELECT Auctions.name FROM Auctions  WHERE Auctions.id = items.auction_id ) as auction_name,
(SELECT catalogs.name FROM catalogs  WHERE catalogs.id = items.catalog_id ) as catalog_name,
title,
hummer_price,
(SELECT commissions.rate FROM commissions JOIN profiles 
	ON profiles.commission_to_sell_id=commissions.id WHERE seller_id = profiles.user_id) as seller_commssion
FROM items JOIN profiles ON seller_id = profiles.user_id WHERE items.seller_id = 13 AND items.is_active = 'sold'
);

SELECT * FROM temp_seller ;




-- ==========================================================================================================================
-- Сбор всех проданных локов конкретного продовца  в инвойс.


DROP PROCEDURE IF EXISTS sp_seller_invoice;
DELIMITER //
CREATE PROCEDURE sp_seller_invoice (
    	
	OUT invoice_result varchar(200))
BEGIN

DECLARE is_end INT DEFAULT 0;

DECLARE item_id			BIGINT;
DECLARE seller_id		BIGINT;
DECLARE seller_name		VARCHAR(255) ;
DECLARE seller_phone		BIGINT;
DECLARE seller_mail		VARCHAR(255);
DECLARE auction_name	VARCHAR(255);
DECLARE catalog_name	VARCHAR(255);
DECLARE title			VARCHAR(255);
DECLARE hummer_price	DECIMAL (11,2);
DECLARE seller_commssion	INT;
DECLARE content	text;
DECLARE cnt int;
DECLARE lots_summ DECIMAL (11,2);
DECLARE is_paid  ENUM('YES', 'NOT');

DECLARE curcat CURSOR FOR SELECT * FROM temp_seller;
DECLARE CONTINUE HANDLER FOR NOT FOUND SET is_end = 1;
SET cnt = 0;
SET lots_summ = 0.00;
SET is_paid = 'NOT';
SET content := '{';
OPEN curcat;
		cycle : LOOP
			FETCH curcat INTO item_id, seller_id, seller_name, seller_phone, seller_mail, auction_name, catalog_name,
			title, hummer_price, seller_commssion;
			#SET invoice_result := invoice_result + 1;
			#SET invoice_result := hummer_price;
			IF is_end THEN LEAVE cycle;
			END IF;
			IF cnt = 0 THEN 
				SET content := concat(content, ' "N',cnt,'" : "', item_id, '", "title',cnt,'" : "', title, '" , "hummer_price',cnt,':"  : "', hummer_price);
				SET cnt = cnt + 1;
				SET lots_summ = lots_summ + hummer_price;
			ELSE
				SET content := concat(content, '",  "N',cnt,'" : "', item_id, '",  "title',cnt,'" : "', title, ' ", "hummer_price',cnt,':"  : "', hummer_price);
				SET cnt = cnt + 1;
				SET lots_summ = lots_summ + hummer_price;
			END IF;
			#call sp_update_item_results(id, @udate_result);
			
		END LOOP cycle;
		SET content := concat(content, '"  }');
		SET lots_summ :=lots_summ*(100-seller_commssion)/100;
		INSERT INTO invoices_for_sellers VALUES (seller_id, lots_summ, content, is_paid);
		SET invoice_result := 'ok';
CLOSE curcat;
END//
DELIMITER ;


call sp_seller_invoice( @invoice_result);
SELECT @invoice_result;
SELECT * FROM invoices_for_sellers ;







-- ==========================================================================================================================
-- Сбор всех инвойсов всех продавцов.
-- Список счетов к оплате для участников продавших лоты.
DROP TABLE IF EXISTS invoices_for_sellers;
CREATE TABLE invoices_for_sellers (
	seller_id BIGINT UNSIGNED NOT NULL,
	to_pay    BIGINT UNSIGNED NOT NULL,
	explntn   JSON,
	is_paid   ENUM('YES', 'NOT'),
	INDEX (seller_id),
	INDEX (is_paid),	
    PRIMARY KEY (seller_id)   
)COMMENT = 'Продавцы'; 




DROP PROCEDURE IF EXISTS sp_sellers_invoices;
DELIMITER //
CREATE PROCEDURE sp_sellers_invoices (
    	
	OUT invoice_result varchar(200))
BEGIN


DECLARE is_end INT DEFAULT 0;
DECLARE id_of_seller BIGINT ;

DECLARE curcat CURSOR FOR SELECT * FROM sellers;
DECLARE CONTINUE HANDLER FOR NOT FOUND SET is_end = 1;

OPEN curcat;
		cycle : LOOP
			FETCH curcat INTO id_of_seller;

			IF is_end THEN LEAVE cycle;
			END IF;
			
		
			DROP TABLE IF EXISTS temp_seller;
			CREATE TEMPORARY TABLE temp_seller (
			SELECT 
			items.id as item_id,
			seller_id as seller_id,
			name as seller_name,
			phone as seller_phone,
			(SELECT users.e_mail FROM users  WHERE seller_id = users.id ) as seller_mail,
			(SELECT Auctions.name FROM Auctions  WHERE Auctions.id = items.auction_id ) as auction_name,
			(SELECT catalogs.name FROM catalogs  WHERE catalogs.id = items.catalog_id ) as catalog_name,
			title,
			hummer_price,
			(SELECT commissions.rate FROM commissions JOIN profiles 
				ON profiles.commission_to_sell_id=commissions.id WHERE seller_id = profiles.user_id) as seller_commssion
			FROM items JOIN profiles ON seller_id = profiles.user_id WHERE items.seller_id = id_of_seller AND items.is_active = 'sold'
			);		
		
		
			
			call sp_seller_invoice( @invoice_result);
			
		END LOOP cycle;

		SET invoice_result := 'ok';
CLOSE curcat;
END//
DELIMITER ;



call sp_sellers_invoices( @invoice_result);
SELECT @invoice_result;
SELECT * FROM invoices_for_sellers ;

-- ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



CREATE OR REPLACE VIEW sold_items AS 

SELECT  
items.is_active,
Auctions.name as Auction_name,
Auctions.day_of_auction as Auction_date, 
catalogs.name as Catalog_name,
items.estimate ,
items.title,
items.desription,
item_conditions.rate as lot_condition,
item_conditions.rate_explanation as lot_explanation,
item_restrictions.restriction_explanation as lot_restriction

FROM items JOIN Auctions ON items.auction_id=Auctions.id 
JOIN catalogs ON items.catalog_id=catalogs.id 
JOIN item_conditions ON items.item_condition=item_conditions.id
JOIN item_restrictions ON items.restriction_id=item_restrictions.id
WHERE items.is_active = 'sold';

SELECT * FROM sold_items;

-- ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


CREATE OR REPLACE VIEW top_5_lots AS 

SELECT  
items.is_active,
Auctions.name as Auction_name,
Auctions.day_of_auction as Auction_date, 
catalogs.name as Catalog_name,
items.hummer_price,
items.title,
items.desription


FROM items JOIN Auctions ON items.auction_id=Auctions.id 
JOIN catalogs ON items.catalog_id=catalogs.id 


WHERE items.is_active = 'sold' ORDER BY items.hummer_price DESC LIMIT 5;

SELECT * FROM top_5_lots;

