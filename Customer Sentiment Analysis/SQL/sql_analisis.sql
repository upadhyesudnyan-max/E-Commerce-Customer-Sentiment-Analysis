-- ============================================
-- E-COMMERCE CUSTOMER SENTIMENT ANALYSIS
-- STEP 2: STORE DATA IN MYSQL
-- ============================================


-- 1. CREATE DATABASE
CREATE DATABASE IF NOT EXISTS ecommerce_sentiment;

-- 2. SELECT DATABASE
USE ecommerce_sentiment;


-- ============================================
-- 3. CREATE PRODUCTS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(255),
    category VARCHAR(100),
    brand VARCHAR(100)
);


-- ============================================
-- 4. CREATE REVIEWS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS reviews (
    review_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    product_id VARCHAR(50),
    review_text TEXT,
    review_date DATE,
    sentiment VARCHAR(30),
    sentiment_score DECIMAL(6,4),

    FOREIGN KEY (product_id)
    REFERENCES products(product_id)
);


-- ============================================
-- 5. CREATE RATINGS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS ratings (
    rating_id INT AUTO_INCREMENT PRIMARY KEY,
    review_id VARCHAR(50),
    customer_id VARCHAR(50),
    product_id VARCHAR(50),
    rating INT,

    FOREIGN KEY (review_id)
    REFERENCES reviews(review_id),

    FOREIGN KEY (product_id)
    REFERENCES products(product_id),

    CHECK (rating BETWEEN 1 AND 5)
);


-- ============================================
-- 6. CREATE CUSTOMER COMPLAINTS TABLE
-- ============================================

CREATE TABLE IF NOT EXISTS customer_complaints (
    complaint_id INT AUTO_INCREMENT PRIMARY KEY,
    review_id VARCHAR(50),
    customer_id VARCHAR(50),
    complaint_type VARCHAR(100),
    complaint_description TEXT,

    FOREIGN KEY (review_id)
    REFERENCES reviews(review_id)
);


-- ============================================
-- 7. CHECK CREATED TABLES
-- ============================================

SHOW TABLES;


-- ============================================
-- 8. CHECK TABLE STRUCTURES
-- ============================================

DESCRIBE products;

DESCRIBE reviews;

DESCRIBE ratings;

DESCRIBE customer_complaints;


-- ============================================
-- 9. CHECK CURRENT DATABASE
-- ============================================

SELECT DATABASE();


-- ============================================
-- 10. CHECK RECORDS
-- ============================================

SELECT * FROM products;

SELECT * FROM reviews;

SELECT * FROM ratings;

SELECT * FROM customer_complaints;


-- ============================================
-- 11. COUNT RECORDS
-- ============================================

SELECT COUNT(*) AS total_products
FROM products;

SELECT COUNT(*) AS total_reviews
FROM reviews;

SELECT COUNT(*) AS total_ratings
FROM ratings;

SELECT COUNT(*) AS total_complaints
FROM customer_complaints;


-- ============================================
-- 12. NEGATIVE REVIEWS
-- ============================================

SELECT
    review_id,
    customer_id,
    product_id,
    review_text,
    sentiment,
    sentiment_score
FROM reviews
WHERE LOWER(sentiment) = 'negative';


-- ============================================
-- 13. NEGATIVE REVIEWS WITH RATINGS
-- ============================================

SELECT
    r.review_id,
    r.customer_id,
    r.product_id,
    r.review_text,
    r.sentiment,
    rt.rating
FROM reviews r
LEFT JOIN ratings rt
    ON r.review_id = rt.review_id
WHERE LOWER(r.sentiment) = 'negative';


-- ============================================
-- 14. LOW-RATED PRODUCTS
-- ============================================

SELECT
    product_id,
    COUNT(*) AS total_ratings,
    ROUND(AVG(rating),2) AS average_rating
FROM ratings
GROUP BY product_id
HAVING AVG(rating) < 3
ORDER BY average_rating ASC;


-- ============================================
-- 15. MOST COMMON NEGATIVE-WORDS
-- ============================================

SELECT
    'bad' AS word,
    COUNT(*) AS frequency
FROM reviews
WHERE LOWER(sentiment) = 'negative'
AND LOWER(review_text) LIKE '%bad%'

UNION ALL

SELECT
    'poor',
    COUNT(*)
FROM reviews
WHERE LOWER(sentiment) = 'negative'
AND LOWER(review_text) LIKE '%poor%'

UNION ALL

SELECT
    'slow',
    COUNT(*)
FROM reviews
WHERE LOWER(sentiment) = 'negative'
AND LOWER(review_text) LIKE '%slow%'

UNION ALL

SELECT
    'problem',
    COUNT(*)
FROM reviews
WHERE LOWER(sentiment) = 'negative'
AND LOWER(review_text) LIKE '%problem%'

UNION ALL

SELECT
    'issue',
    COUNT(*)
FROM reviews
WHERE LOWER(sentiment) = 'negative'
AND LOWER(review_text) LIKE '%issue%'

UNION ALL

SELECT
    'quality',
    COUNT(*)
FROM reviews
WHERE LOWER(sentiment) = 'negative'
AND LOWER(review_text) LIKE '%quality%'

UNION ALL

SELECT
    'service',
    COUNT(*)
FROM reviews
WHERE LOWER(sentiment) = 'negative'
AND LOWER(review_text) LIKE '%service%'

UNION ALL

SELECT
    'delivery',
    COUNT(*)
FROM reviews
WHERE LOWER(sentiment) = 'negative'
AND LOWER(review_text) LIKE '%delivery%'

UNION ALL

SELECT
    'damaged',
    COUNT(*)
FROM reviews
WHERE LOWER(sentiment) = 'negative'
AND LOWER(review_text) LIKE '%damaged%'

UNION ALL

SELECT
    'refund',
    COUNT(*)
FROM reviews
WHERE LOWER(sentiment) = 'negative'
AND LOWER(review_text) LIKE '%refund%'

ORDER BY frequency DESC;

show tables;
select * from products;