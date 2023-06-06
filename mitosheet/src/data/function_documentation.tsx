
/*
    WARNING: THIS FILE IS AUTOGENERATED. ANY EDITS YOU MAKE WILL BE OVERWRITTEN!
*/

export interface FunctionDocumentationObject {
    function: string;
    description: string;
    search_terms: string[];
    examples?: (string)[] | null;
    syntax: string;
    syntax_elements?: (SyntaxElementsEntity)[] | null;
}

export interface SyntaxElementsEntity {
    element: string;
    description: string;
}

export const functionDocumentationObjects: FunctionDocumentationObject[] = [{"function": "ABS", "description": "Returns the absolute value of the passed number or series.", "search_terms": ["abs", "absolute value"], "examples": ["ABS(-1.3)", "ABS(A)"], "syntax": "ABS(value)", "syntax_elements": [{"element": "value", "description": "The value or series to take the absolute value of."}]}, {"function": "AND", "description": "Returns True if all of the provided arguments are True, and False if any of the provided arguments are False.", "search_terms": ["and", "&", "if", "conditional"], "examples": ["AND(True, False)", "AND(Nums > 100, Nums < 200)", "AND(Pay > 10, Pay < 20, Status == 'active')"], "syntax": "AND(boolean_condition1, [boolean_condition2, ...])", "syntax_elements": [{"element": "boolean_condition1", "description": "An expression or series that returns True or False values. See IF documentation for a list of conditons."}, {"element": "boolean_condition2 ... [OPTIONAL]", "description": "An expression or series that returns True or False values. See IF documentation for a list of conditons."}]}, {"function": "AVG", "description": "Returns the numerical mean value of the passed numbers and series.", "search_terms": ["avg", "average", "mean"], "examples": ["AVG(1, 2)", "AVG(A, B)", "AVG(A, 2)"], "syntax": "AVG(value1, [value2, ...])", "syntax_elements": [{"element": "value1", "description": "The first number or series to consider when calculating the average."}, {"element": "value2, ... [OPTIONAL]", "description": "Additional numbers or series to consider when calculating the average."}]}, {"function": "BOOL", "description": "Converts the passed arguments to boolean values, either True or False. For numberic values, 0 converts to False while all other values convert to True.", "search_terms": ["bool", "boolean", "true", "false", "dtype", "convert"], "examples": ["BOOL(Amount_Payed)", "AND(BOOL(Amount_Payed), Is_Paying)"], "syntax": "BOOL(series)", "syntax_elements": [{"element": "series", "description": "An series to convert to boolean values, either True or False."}]}, {"function": "CLEAN", "description": "Returns the text with the non-printable ASCII characters removed.", "search_terms": ["clean", "trim", "remove"], "examples": ["CLEAN(A)"], "syntax": "CLEAN(string)", "syntax_elements": [{"element": "string", "description": "The string or series whose non-printable characters are to be removed."}]}, {"function": "CONCAT", "description": "Returns the passed strings and series appended together.", "search_terms": ["&", "concatenate", "append", "combine"], "examples": ["CONCAT('Bite', 'the bullet')", "CONCAT(A, B)"], "syntax": "CONCAT(string1, [string2, ...])", "syntax_elements": [{"element": "string1", "description": "The first string or series."}, {"element": "string2, ... [OPTIONAL]", "description": "Additional strings or series to append in sequence."}]}, {"function": "CORR", "description": "Computes the correlation between two series, excluding missing values.", "search_terms": ["corr", "correlation", "r^2"], "examples": ["=CORR(A, B)", "=CORR(B, A)"], "syntax": "CORR(series_one, series_two)", "syntax_elements": [{"element": "series_one", "description": "The number series to convert to calculate the correlation."}, {"element": "series_two", "description": "The number series to convert to calculate the correlation."}]}, {"function": "DATEVALUE", "description": "Converts a given string to a date series.", "search_terms": ["datevalue", "date value", "date", "string to date", "datetime", "dtype", "convert"], "examples": ["DATEVALUE(date_column)", "DATEVALUE('2012-12-22')"], "syntax": "DATEVALUE(date_string)", "syntax_elements": [{"element": "date_string", "description": "The date string to turn into a date object."}]}, {"function": "DAY", "description": "Returns the day of the month that a specific date falls on, as a number.", "search_terms": ["day", "date"], "examples": ["DAY(date_column)", "DAY('2012-12-22')"], "syntax": "DAY(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the day of."}]}, {"function": "ENDOFBUSINESSMONTH", "description": "Given a date, returns the end of the buisness month. E.g. the last weekday.", "search_terms": ["business", "month", "eom", "eobm", "date", "workday", "end"], "examples": ["ENDOFBUSINESSMONTH(date_column)", "ENDOFBUSINESSMONTH('2012-12-22')"], "syntax": "ENDOFBUSINESSMONTH(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the end of the business month of."}]}, {"function": "ENDOFMONTH", "description": "Given a date, returns the end of the month, as a date. E.g. input of 12-22-1997 will return 12-31-1997.", "search_terms": ["month", "eom", "date", "workday", "end", "eomonth"], "examples": ["ENDOFMONTH(date_column)", "ENDOFMONTH('2012-12-22')"], "syntax": "ENDOFMONTH(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the last day of the month of."}]}, {"function": "EXP", "description": "Returns e, the base of the natural logarithm, raised to the power of passed series.", "search_terms": ["exp", "exponent", "log", "natural log"], "examples": ["=EXP(data)", "=EXP(A)"], "syntax": "EXP(series)", "syntax_elements": [{"element": "series", "description": "The series to raise e to."}]}, {"function": "FILLNAN", "description": "Replaces the NaN values in the series with the replacement value.", "search_terms": ["fillnan", "nan", "fill nan", "missing values", "null", "null value", "fill null"], "examples": ["FILLNAN(A, 10)", "FILLNAN(A, 'replacement')"], "syntax": "FILLNAN(series, replacement)", "syntax_elements": [{"element": "series", "description": "The series to replace the NaN values in."}, {"element": "replacement", "description": "A string, number, or date to replace the NaNs with."}]}, {"function": "FIND", "description": "Returns the position at which a string is first found within text, case-sensitive. Returns 0 if not found.", "search_terms": ["find", "search"], "examples": ["FIND(A, 'Jack')", "FIND('Ben has a friend Jack', 'Jack')"], "syntax": "FIND(text_to_search, search_for)", "syntax_elements": [{"element": "text_to_search", "description": "The text or series to search for the first occurrence of search_for."}, {"element": "search_for", "description": "The string to look for within text_to_search."}]}, {"function": "FLOAT", "description": "Converts a string series to a float series. Any values that fail to convert will return NaN.", "search_terms": ["number", "to number"], "examples": ["=FLOAT(Prices_string)", "=FLOAT('123.123')"], "syntax": "FLOAT(string_series)", "syntax_elements": [{"element": "string_series", "description": "The series or string to convert to a float."}]}, {"function": "GETNEXTVALUE", "description": "Returns the next value from series that meets the condition.", "search_terms": ["ffill"], "examples": ["GETNEXTVALUE(Max_Balances, Max_Balances > 0)"], "syntax": "GETNEXTVALUE(series, condition)", "syntax_elements": [{"element": "series", "description": "The series to get the next value from."}, {"element": "condition", "description": "When condition is True, a new previous value is set, and carried backwards until the condition is True again."}]}, {"function": "GETPREVIOUSVALUE", "description": "Returns the value from series that meets the condition.", "search_terms": ["ffill"], "examples": ["GETPREVIOUSVALUE(Max_Balances, Max_Balances > 0)"], "syntax": "GETPREVIOUSVALUE(series, condition)", "syntax_elements": [{"element": "series", "description": "The series to get the previous value from."}, {"element": "condition", "description": "When condition is True, a new previous value is set, and carried forward until the condition is True again."}]}, {"function": "HOUR", "description": "Returns the hour component of a specific date, as a number.", "search_terms": ["hour", "hr"], "examples": ["HOUR(date_column)", "HOUR('2012-12-22 09:45:00')"], "syntax": "HOUR(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the hour of."}]}, {"function": "IF", "description": "Returns one value if the condition is True. Returns the other value if the conditon is False.", "search_terms": ["if", "conditional", "and", "or"], "examples": ["IF(Status == 'success', 1, 0)", "IF(Nums > 100, 100, Nums)", "IF(AND(Grade >= .6, Status == 'active'), 'pass', 'fail')"], "syntax": "IF(boolean_condition, value_if_true, value_if_false)", "syntax_elements": [{"element": "boolean_condition", "description": "An expression or series that returns True or False values. Valid conditions for comparison include ==, !=, >, <, >=, <=."}, {"element": "value_if_true", "description": "The value the function returns if condition is True."}, {"element": "value_if_false", "description": "The value the function returns if condition is False."}]}, {"function": "INT", "description": "Converts a string series to a int series. Any values that fail to convert will return 0.", "search_terms": ["number", "to integer"], "examples": ["=INT(Prices_string)", "=INT('123')"], "syntax": "INT(string_series)", "syntax_elements": [{"element": "string_series", "description": "The series or string to convert to a int."}]}, {"function": "KURT", "description": "Computes the unbiased kurtosis, a measure of tailedness, of a series, excluding missing values.", "search_terms": ["kurtosis"], "examples": ["=KURT(A)", "=KURT(A * B)"], "syntax": "KURT(series)", "syntax_elements": [{"element": "series", "description": "The series to calculate the unbiased kurtosis of."}]}, {"function": "LEFT", "description": "Returns a substring from the beginning of a specified string.", "search_terms": ["left"], "examples": ["LEFT(A, 2)", "LEFT('The first character!')"], "syntax": "LEFT(string, [number_of_characters])", "syntax_elements": [{"element": "string", "description": "The string or series from which the left portion will be returned."}, {"element": "number_of_characters [OPTIONAL, 1 by default]", "description": "The number of characters to return from the start of string."}]}, {"function": "LEN", "description": "Returns the length of a string.", "search_terms": ["length", "size"], "examples": ["LEN(A)", "LEN('This is 21 characters')"], "syntax": "LEN(string)", "syntax_elements": [{"element": "string", "description": "The string or series whose length will be returned."}]}, {"function": "LOG", "description": "Calculates the logarithm of the passed series with an optional base.", "search_terms": ["log", "logarithm", "natural log"], "examples": ["LOG(10) = 1", "LOG(100, 10) = 2"], "syntax": "LOG(series, [base])", "syntax_elements": [{"element": "series", "description": "The series to take the logarithm of."}, {"element": "base [OPTIONAL]", "description": "The base of the logarithm to use. Defaults to 10 if no base is passed."}]}, {"function": "LOWER", "description": "Converts a given string to lowercase.", "search_terms": ["lowercase", "uppercase"], "examples": ["=LOWER('ABC')", "=LOWER(A)", "=LOWER('Nate Rush')"], "syntax": "LOWER(string)", "syntax_elements": [{"element": "string", "description": "The string or series to convert to lowercase."}]}, {"function": "MAX", "description": "Returns the maximum value among the passed arguments.", "search_terms": ["max", "maximum", "minimum"], "examples": ["MAX(10, 11)", "MAX(Old_Data, New_Data)"], "syntax": "MAX(value1, [value2, ...])", "syntax_elements": [{"element": "value1", "description": "The first number or column to consider for the maximum value."}, {"element": "value2, ... [OPTIONAL]", "description": "Additional numbers or columns to compute the maximum value from."}]}, {"function": "MID", "description": "Returns a segment of a string.", "search_terms": ["middle"], "examples": ["MID(A, 2, 2)", "MID('Some middle characters!', 3, 4)"], "syntax": "MID(string, starting_at, extract_length)", "syntax_elements": [{"element": "string", "description": "The string or series to extract the segment from."}, {"element": "starting_at", "description": "The index from the left of string from which to begin extracting."}, {"element": "extract_length", "description": "The length of the segment to extract."}]}, {"function": "MIN", "description": "Returns the minimum value among the passed arguments.", "search_terms": ["min", "minimum", "maximum"], "examples": ["MIN(10, 11)", "MIN(Old_Data, New_Data)"], "syntax": "MIN(value1, [value2, ...])", "syntax_elements": [{"element": "value1", "description": "The first number or column to consider for the minumum value."}, {"element": "value2, ... [OPTIONAL]", "description": "Additional numbers or columns to compute the minumum value from."}]}, {"function": "MINUTE", "description": "Returns the minute component of a specific date, as a number.", "search_terms": ["minute", "min"], "examples": ["MINUTE(date_column)", "MINUTE('2012-12-22 09:45:00')"], "syntax": "MINUTE(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the minute of."}]}, {"function": "MONTH", "description": "Returns the month that a specific date falls in, as a number.", "search_terms": ["month", "date"], "examples": ["MONTH(date_column)", "MONTH('2012-12-22')"], "syntax": "MONTH(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the month of."}]}, {"function": "MULTIPLY", "description": "Returns the product of two numbers.", "search_terms": ["mulitply", "product"], "examples": ["MULTIPLY(2,3)", "MULTIPLY(A,3)"], "syntax": "MULTIPLY(factor1, [factor2, ...])", "syntax_elements": [{"element": "factor1", "description": "The first number to multiply."}, {"element": "factor2, ... [OPTIONAL]", "description": "Additional numbers or series to multiply."}]}, {"function": "OR", "description": "Returns True if any of the provided arguments are True, and False if all of the provided arguments are False.", "search_terms": ["or", "if", "conditional"], "examples": ["OR(True, False)", "OR(Status == 'success', Status == 'pass', Status == 'passed')"], "syntax": "OR(boolean_condition1, [boolean_condition2, ...])", "syntax_elements": [{"element": "boolean_condition1", "description": "An expression or series that returns True or False values. See IF documentation for a list of conditons."}, {"element": "boolean_condition2 ... [OPTIONAL]", "description": "An expression or series that returns True or False values. See IF documentation for a list of conditons."}]}, {"function": "POWER", "description": "The POWER function can be used to raise a number to a given power.", "search_terms": ["power", "raise", "exponent", "square", "cube"], "examples": ["POWER(4, 1/2)", "POWER(Dose, 2)"], "syntax": "POWER(value, exponent)", "syntax_elements": [{"element": "value", "description": "Number to raise to a power."}, {"element": "exponent", "description": "The number to raise value to."}]}, {"function": "PROPER", "description": "Capitalizes the first letter of each word in a specified string.", "search_terms": ["proper", "capitalize"], "examples": ["=PROPER('nate nush')", "=PROPER(A)"], "syntax": "PROPER(string)", "syntax_elements": [{"element": "string", "description": "The value or series to convert to convert to proper case."}]}, {"function": "QUARTER", "description": "Returns the quarter (1-4) that a specific date falls in, as a number.", "search_terms": ["quarter"], "examples": ["QUARTER(date_column)", "QUARTER('2012-12-22')"], "syntax": "QUARTER(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the quarter of."}]}, {"function": "RIGHT", "description": "Returns a substring from the beginning of a specified string.", "search_terms": [], "examples": ["RIGHT(A, 2)", "RIGHT('The last character!')"], "syntax": "RIGHT(string, [number_of_characters])", "syntax_elements": [{"element": "string", "description": "The string or series from which the right portion will be returned."}, {"element": "number_of_characters [OPTIONAL, 1 by default]", "description": "The number of characters to return from the end of string."}]}, {"function": "ROUND", "description": "Rounds a number to a given number of decimals.", "search_terms": ["round", "decimal", "integer"], "examples": ["ROUND(1.3)", "ROUND(A, 2)"], "syntax": "ROUND(value, [decimals])", "syntax_elements": [{"element": "value", "description": "The value or series to round."}, {"element": "decimals", "description": " The number of decimals to round to. Default is 0."}]}, {"function": "SECOND", "description": "Returns the seconds component of a specific date, as a number.", "search_terms": ["second", "sec"], "examples": ["SECOND(date_column)", "SECOND('2012-12-22 09:23:05')"], "syntax": "SECOND(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the seconds of."}]}, {"function": "SKEW", "description": "Computes the skew of a series, excluding missing values.", "search_terms": [], "examples": ["=SKEW(A)", "=SKEW(A * B)"], "syntax": "SKEW(series)", "syntax_elements": [{"element": "series", "description": "The series to calculate the skew of."}]}, {"function": "STARTOFBUSINESSMONTH", "description": "Given a date, returns the most recent start of the business month, as a state. E.g. the first weekday.", "search_terms": ["business", "month", "SOM", "SOBM", "date", "start"], "examples": ["STARTOFBUSINESSMONTH(date_column)", "STARTOFBUSINESSMONTH('2012-12-22 09:23:05')"], "syntax": "STARTOFBUSINESSMONTH(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the most recent beginning of month business day of."}]}, {"function": "STARTOFMONTH", "description": "Given a date, returns the start of the month, as a date. E.g. input of 12-22-1997 will return 12-1-1997.", "search_terms": ["month", "SOM", "date", "start"], "examples": ["STARTOFMONTH(date_column)", "STARTOFMONTH('2012-12-22 09:23:05')"], "syntax": "STARTOFMONTH(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the first day of the month of."}]}, {"function": "STDEV", "description": "Computes the standard deviation of a series, excluding missing values.", "search_terms": ["standard", "deviation", "standard", "distribution"], "examples": ["=STDEV(A)", "=STDEV(A * B)"], "syntax": "STDEV(series)", "syntax_elements": [{"element": "series", "description": "The series to calculate the standard deviation of."}]}, {"function": "STRIPTIMETODAYS", "description": "Returns the date with a seconds, minutes, and hours component of 00:00:00.", "search_terms": ["time", "date", "days", "strip"], "examples": ["STRIPTIMETODAYS(date_column)", "STRIPTIMETODAYS('2012-12-22 09:23:05')"], "syntax": "STRIPTIMETODAYS(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to reset the seconds, minutes, and hours component of."}]}, {"function": "STRIPTIMETOHOURS", "description": "Returns the date with a seconds and minutes component of 00:00.", "search_terms": ["time", "date", "hours", "strip"], "examples": ["STRIPTIMETOHOURS(date_column)", "STRIPTIMETOHOURS('2012-12-22 09:23:05')"], "syntax": "STRIPTIMETOHOURS(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to reset the seconds and minutes component of."}]}, {"function": "STRIPTIMETOMINUTES", "description": "Returns the date with a seconds component of 00.", "search_terms": ["time", "date", "minutes", "strip"], "examples": ["STRIPTIMETOMINUTES(date_column)", "STRIPTIMETOMINUTES('2012-12-22 09:23:05')"], "syntax": "STRIPTIMETOMINUTES(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to reset the seconds component of."}]}, {"function": "STRIPTIMETOMONTHS", "description": "Returns the date adjusted to the start of the month.", "search_terms": ["time", "date", "months", "strip"], "examples": ["STRIPTIMETOMONTHS(date_column)", "STRIPTIMETOMONTHS('2012-12-22 09:23:05')"], "syntax": "STRIPTIMETOMONTHS(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to reset the seconds, minutes, hours, and days of."}]}, {"function": "STRIPTIMETOYEARS", "description": "Returns the date adjusted to the start of the year.", "search_terms": ["time", "date", "years", "strip"], "examples": ["STRIPTIMETOYEARS(date_column)", "STRIPTIMETOYEARS('2012-12-22 09:23:05')"], "syntax": "STRIPTIMETOYEARS(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to reset the seconds, minutes, hours, days, and month components of."}]}, {"function": "SUBSTITUTE", "description": "Replaces existing text with new text in a string.", "search_terms": ["replace", "find and replace"], "examples": ["SUBSTITUTE('Better great than never', 'great', 'late')", "SUBSTITUTE(A, 'dog', 'cat')"], "syntax": "SUBSTITUTE(text_to_search, search_for, replace_with, [count])", "syntax_elements": [{"element": "text_to_search", "description": "The text within which to search and replace."}, {"element": "search_for", "description": " The string to search for within text_to_search."}, {"element": "replace_with", "description": "The string that will replace search_for."}, {"element": "count", "description": "The number of times to perform the substitute. Default is all."}]}, {"function": "SUM", "description": "Returns the sum of the given numbers and series.", "search_terms": ["add"], "examples": ["SUM(10, 11)", "SUM(A, B, D, F)", "SUM(A, B, D, F)"], "syntax": "SUM(value1, [value2, ...])", "syntax_elements": [{"element": "value1", "description": "The first number or column to add together."}, {"element": "value2, ... [OPTIONAL]", "description": "Additional numbers or columns to sum."}]}, {"function": "SUMPRODUCT", "description": "Returns the sum of the product of the passed arguments.", "search_terms": ["sum product", "sumproduct", "sum", "product", "weighted average"], "examples": ["SUMPRODUCT(A:A, B:B)", "SUMPRODUCT(A:B)"], "syntax": "SUMPRODUCT(array1, [array2, ...])", "syntax_elements": [{"element": "array1", "description": "The first array argument whose components you want to multiply and then add."}, {"element": "value2, ... [OPTIONAL]", "description": "Additional series to multiply then add."}]}, {"function": "TEXT", "description": "Turns the passed series into a string.", "search_terms": ["string", "dtype"], "examples": ["=TEXT(Product_Number)", "=TEXT(Start_Date)"], "syntax": "TEXT(series)", "syntax_elements": [{"element": "series", "description": "The series to convert to a string."}]}, {"function": "TRIM", "description": "Returns a string with the leading and trailing whitespace removed.", "search_terms": ["trim", "whitespace", "spaces"], "examples": ["=TRIM('  ABC')", "=TRIM('  ABC  ')", "=TRIM(A)"], "syntax": "TRIM(string)", "syntax_elements": [{"element": "string", "description": "The value or series to remove the leading and trailing whitespace from."}]}, {"function": "TYPE", "description": "Returns the type of each element of the passed series. Return values are 'number', 'str', 'bool', 'datetime', 'object', or 'NaN'.", "search_terms": ["type", "dtype"], "examples": ["TYPE(Nums_and_Strings)", "IF(TYPE(Account_Numbers) != 'NaN', Account_Numbers, 0)"], "syntax": "TYPE(series)", "syntax_elements": [{"element": "series", "description": "The series to get the type of each element of."}]}, {"function": "UPPER", "description": "Converts a given string to uppercase.", "search_terms": ["uppercase", "capitalize"], "examples": ["=UPPER('abc')", "=UPPER(A)", "=UPPER('Nate Rush')"], "syntax": "UPPER(string)", "syntax_elements": [{"element": "string", "description": "The string or series to convert to uppercase."}]}, {"function": "VALUE", "description": "Converts a string series to a number series. Any values that fail to convert will return an NaN.", "search_terms": ["number", "to number", "dtype", "convert", "parse"], "examples": ["=VALUE(A)", "=VALUE('123')"], "syntax": "VALUE(string)", "syntax_elements": [{"element": "string", "description": "The string or series to convert to a number."}]}, {"function": "VAR", "description": "Computes the variance of a series, excluding missing values.", "search_terms": ["variance"], "examples": ["=VAR(A)", "=VAR(A - B)"], "syntax": "VAR(series)", "syntax_elements": [{"element": "series", "description": "The series to calculate the variance of."}]}, {"function": "WEEK", "description": "Returns the week (1-52) of a specific date, as a number.", "search_terms": ["week", "1", "52"], "examples": ["WEEK(date_column)", "WEEK('2012-12-22 09:23:05')"], "syntax": "WEEK(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the week of."}]}, {"function": "WEEKDAY", "description": "Returns the day of the week that a specific date falls on. 1-7 corresponds to Monday-Sunday.", "search_terms": ["weekday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"], "examples": ["WEEKDAY(date_column)", "WEEKDAY('2012-12-22')"], "syntax": "WEEKDAY(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the weekday of."}]}, {"function": "YEAR", "description": "Returns the day of the year that a specific date falls in, as a number.", "search_terms": ["year", "date"], "examples": ["YEAR(date_column)", "YEAR('2012-12-22')"], "syntax": "YEAR(date)", "syntax_elements": [{"element": "date", "description": "The date or date series to get the month of."}]}]