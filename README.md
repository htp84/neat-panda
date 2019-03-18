
#Neat Panda

[![pypi](https://img.shields.io/pypi/v/neat_panda.svg)](https://pypi.python.org/pypi/neat_panda)

[![Build status](https://img.shields.io/travis/htp84/neat_panda.svg)](https://travis-ci.org/htp84/neat_panda)

[![Coverage](https://codecov.io/github/htp84/neat_panda/coverage.svg?branch=master)](https://codecov.io/gh/htp84/neat-panda)

Neat Panda contains functions written to mimic the R package Tidyr.


* Free software: MIT license


## Features
### Spread
#### R
```R
library(tidyr)
library(dplyr)
library(gapminder)

gapminder2 <- gapminder %>% select(country, continent, year, pop)
gapminder3 <- gapminder2 %>% spread(key=year, value=pop)
head(gapminder3, n = 5)
```
##### Output
```
# A tibble: 5 x 14
  country     continent   `1952`   `1957`   `1962`   `1967`   `1972`   `1977`   `1982`   `1987`   `1992`   `1997`   `2002`   `2007`
  <fct>       <fct>        <int>    <int>    <int>    <int>    <int>    <int>    <int>    <int>    <int>    <int>    <int>    <int>
1 Afghanistan Asia       8425333  9240934 10267083 11537966 13079460 14880372 12881816 13867957 16317921 22227415 25268405 31889923
2 Albania     Europe     1282697  1476505  1728137  1984060  2263554  2509048  2780097  3075321  3326498  3428038  3508512  3600523
3 Algeria     Africa     9279525 10270856 11000948 12760499 14760787 17152804 20033753 23254956 26298373 29072015 31287142 33333216
4 Angola      Africa     4232095  4561361  4826015  5247469  5894858  6162675  7016384  7874230  8735988  9875024 10866106 12420476
5 Argentina   Americas  17876956 19610538 21283783 22934225 24779799 26983828 29341374 31620918 33958947 36203463 38331121 40301927
```
#### Python
```python
from neat_panda import spread
from gapminder import gapminder

gapminder2 = gapminder[["country", "continent", "year", "pop"]]
gapminder3 = spread(df=gapminder2, key="year", value="pop")
# or
gapminder3 = gapminder2.pipe(spread, key="year", value="pop")

gapminder3.head()
```
##### Output
```
       country continent      1952      1957      1962      1967      1972      1977      1982      1987      1992      1997      2002      2007
0  Afghanistan      Asia   8425333   9240934  10267083  11537966  13079460  14880372  12881816  13867957  16317921  22227415  25268405  31889923
1      Albania    Europe   1282697   1476505   1728137   1984060   2263554   2509048   2780097   3075321   3326498   3428038   3508512   3600523
2      Algeria    Africa   9279525  10270856  11000948  12760499  14760787  17152804  20033753  23254956  26298373  29072015  31287142  33333216
3       Angola    Africa   4232095   4561361   4826015   5247469   5894858   6162675   7016384   7874230   8735988   9875024  10866106  12420476
4    Argentina  Americas  17876956  19610538  21283783  22934225  24779799  26983828  29341374  31620918  33958947  36203463  38331121  40301927
```


### Gather
#### R
```R
library(tidyr)
# hitta bra exempeldata
```
#### Python
```python
from neat_panda import gather
# hitta bra exempeldata
```





