// Import necessary React and Next.js modules and components
import React from 'react';
import Head from 'next/head';
import Image from "next/image"
import Link from "next/link"


import pageStyles from '../../styles/Page.module.css';
import excelToPythonStyles from '../../styles/ExcelToPython.module.css';
import titleStyles from '../../styles/Title.module.css'
import textImageSplitStyles from '../../styles/TextImageSplit.module.css'

import { classNames } from '../../utils/classNames';
import TextButton from '../../components/TextButton/TextButton';
import Header from '../../components/Header/Header';
import Footer from '../../components/Footer/Footer';

import CodeBlock from '../../components/CodeBlock/CodeBlock';
import GlossayHorizontalNavbar from '../../components/Glossary/HorizontalNav/HorizontalNav';
import HorizontalNavItem from '../../components/Glossary/HorizontalNavItem/HorizontalNavItem';
import CTAButtons from '../../components/CTAButtons/CTAButtons';


// Define the specific function name
const functionName = "ABS";

// Define your page content and metadata
const PageContent = () => {

  return (
    <>
      <Head>
        {/* Title Tag */}
        <title>{`Excel to Python: ${functionName} - A Complete Guide`}</title>
        
        {/* Meta Description */}
        <meta
          name="description"
          content={`Learn how to convert Excel's ${functionName} function to Python using Pandas. This comprehensive guide provides step-by-step instructions and practical examples.`}
        />
        
        {/* Canonical URL (if applicable) */}
        {/* <link rel="canonical" href={`https://www.example.com/excel-to-python/${function-name}-guide`} /> */}
        
        {/* Open Graph Tags (for social media sharing) */}
        <meta
          property="og:title"
          content={`Excel to Python: ${functionName} - A Complete Guide`}
        />
        <meta
          property="og:description"
          content={`Learn how to convert Excel's ${functionName} function to Python using Pandas. This comprehensive guide provides step-by-step instructions and practical examples.`}
        />
        {/* Add more Open Graph tags as needed */}
        
        {/* Twitter Card Tags (for Twitter sharing) */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta
          name="twitter:title"
          content={`Excel to Python: ${functionName} - A Complete Guide`}
        />
        <meta
          name="twitter:description"
          content={`Learn how to convert Excel's ${functionName} function to Python using Pandas. This comprehensive guide provides step-by-step instructions and practical examples.`}
        />
        {/* Add more Twitter Card tags as needed */}
        
        {/* Other SEO-related tags (structured data, robots meta, etc.) */}
        {/* Add other SEO-related tags here */}
      </Head>
      <Header/>

      <div className={pageStyles.container}>
        <main className={classNames(pageStyles.main, excelToPythonStyles.main)}>
          <section className={classNames(excelToPythonStyles.title_card, excelToPythonStyles.section)}>
            <div className={excelToPythonStyles.horizontal_navbar_container}>
              <GlossayHorizontalNavbar>
                <HorizontalNavItem title="functions" href='/spreadsheet-automation'/>
                <HorizontalNavItem title="abs" href='/spreadsheet-automation'/>
              </GlossayHorizontalNavbar>
            </div>
            
            <h1>How to Implement Excels: <span className='text-highlight'>{functionName}</span> function in Pandas</h1>
            <div className={classNames(excelToPythonStyles.related_functions_card)}>
              <p>Related Functions</p>
              <TextButton 
                text="SUM"
                variant='white'
                fontSize='small'
              />
              <TextButton
                text='ROUND'
                variant='white'
                fontSize='small'
              />
              <TextButton
                text='CEIL'
                variant='white'
                fontSize='small'
              />
            </div>
          </section>

          
          <section className={excelToPythonStyles.section}>
            <p>
              Excel&apos;s ABS function finds the absolute value of a number. The absolute value of a function is the non-negative value of a number. The absolute value function is commonly used, for example, to calculate the distance between two points. Regardless of the order we look at the points, the distance should always be positive.
            </p>
            <p>
              This page explains how to implement Excel&apos;s ABS function in Python, so you can automate Excel reports using Python and Pandas.
            </p>
          </section>

          {/* Understanding the Excel Function */}
          <section className={excelToPythonStyles.section}>
              <h2>Understanding the Excel Function</h2>
              <p>
                The ABS function in Excel takes a single parameters and returns its absolute value.
              </p>
              <p>
                =ABS(<span className='text-highlight'>number</span>)
              </p>
              <h3 className={excelToPythonStyles.h3}>ABS Excel Syntax</h3>
              <table className={excelToPythonStyles.table}>
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Description</th>
                    <th>Data Type</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>number</td>
                    <td>The number you want to take the absolute value of</td>
                    <td>number</td>
                  </tr>
                </tbody>
              </table>

              <h3 className={excelToPythonStyles.h3}>ABS Examples</h3>
              <table className={excelToPythonStyles.table}>
                <thead>
                  <tr>
                    <th>Formula</th>
                    <th>Description</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>=ABS(-5)</td>
                    <td>Calculate the absolute value of -5</td>
                    <td>5</td>
                  </tr>
                  <tr>
                    <td>=ABS(2*-2)</td>
                    <td>Calculate the absolute value of 2 * -2</td>
                    <td>4</td>
                  </tr>
                </tbody>
              </table>
          </section>

          {/* Equivalent Python Code Using Pandas */}
          <section className={excelToPythonStyles.section}>
              <h2>Implementing the Absolute Value function in Pandas</h2>
              <p>
                To replicate the ABS function in Excel using Python and Pandas, you can use the `abs()` function available in Pandas. Below are examples of how to achieve the same functionality.
              </p>
              
              <h3 className={excelToPythonStyles.h3}>Calculate the absolute value for every cell in a Pandas series</h3>
              <p>
                The most common way to use the function in Excel is to apply it directly to a column or series of numbers in a Pandas DataFrame.
              </p>

              <CodeBlock code={`# Calculate the absolute value of the Numbers column
df['ABS_Result'] = df['Numbers'].abs()`}
              />
              <h3 className={excelToPythonStyles.h3}>Finding the absolute difference between two columns</h3>
              <p>
                To use the absolute value as part of a more complex operation, you can use the `apply()` function to apply the operation to every element in an pandas dataframe column.
              </p>
              <CodeBlock code = {`# Calculate the absolute difference between Column1 and Column2
df['Absolute_Difference'] = (df['Column1'] - df['Column2']).abs()`
              }/>
              <h3 className={excelToPythonStyles.h3}>Using ABS as part of a more complex operation</h3>
              <p>
                To use the absolute value as part of a more complex operation, you can use the `apply()` function to apply the operation to every element in an pandas dataframe column.
              </p>
              <CodeBlock code = {`# Define a function to calculate the absolute sum of a row
def abs_sum(row):
  return row.abs().sum()
                
# Create a new column 'ABS_SUM' by applying the custom function 
df['ABS_SUM'] = df.['ABS'].abs(), axis=1)`
              }/>
          </section>

          {/* Common Pitfalls and Tips */}
          <section className={excelToPythonStyles.section}>
            <h2>Common mistakes when implementing the ABS function in Python</h2>
            <p>
              When implementing the ABS function in Python, there are a few common challenges that you might run into.
            </p>
            <h3 className={excelToPythonStyles.h3}>Handling Missing Values</h3>
            <p>
              If you execute the ABS value function on a cell that contains new data in Excel, it will simply return 0. However, in Pandas, empty cells are represented by the Python NoneType. Using the .abs() function on the NoneType will create this error <code>`TypeError: bad operand type for abs(): 'NoneType'`</code>.
            </p>
            <p>
              To resolve this error, before calling the absolute value function, use the fillnan function to replace all missing values with 0. Doing so will make your absolute value function handle missing values exactly the same as Excel.
            </p>
            <CodeBlock code={`# Fill missing values with 0 so it is handled the same was as Excel
df.fillna(0, inplace=True)

# Calculate the absolute value
df['ABS_SUM'] = df['A'].abs()`}/>          
              <h3 className={excelToPythonStyles.h3}>Handling non-numeric values</h3>
              <p>
                In Python, when you use the ABS function you don't have to think about the data types of the input numbers. In fact, most of the time you never have to think about the datatypes of your data in Excel. However, in Python, each column has an explicit data type and each function exepcts a specific data type as the input.
              </p>
              <p>
                Python's .abs function expects the input to be an int (integer) or float (number with decimals). Before calling the .abs function you can make sure that the input is the correct dtype using Pandas .astype formula.
              </p>
              <CodeBlock code={`# Convert the columns to numeric data types (float)
df[A] = df['A'].astype(float)

# Then, replace any cell that could not be converted to a float
# with the value 0, so it’s handled the same as Excel.
df.fillna(0, inplace=True)

# Calculate the absolute value
df['ABS_SUM'] = df['A'].abs()`}/>
          </section>

          <section className={pageStyles.background_card}>
            <div>
              <h2 className={titleStyles.title}>
                Don&apos;t want to re-implement Excel&apos;s functionality in Python?
              </h2>
              <div className='center'>
                  <CTAButtons variant='download' align='center'/>
              </div> 
            </div>
            <div className={classNames(pageStyles.subsection, pageStyles.subsection_justify_baseline)}>
              <div className={textImageSplitStyles.functionality_text}>
                <h2>
                  <span className='text-highlight'>Edit a spreadsheet.</span> <br></br>
                  Generate Python.
                </h2>
                <p>
                  Mito is the easiest way to write Excel formulas in Python. 
                  Every edit you make in the Mito spreadsheet is automatically converted to Python code.
                </p>
                <a href="https://docs.trymito.io/how-to/importing-data-to-mito" target="_blank" rel="noreferrer" className={pageStyles.link_with_p_tag_margins}>
                  View all 100+ transformations →
                </a>
              </div>
              <div className={classNames(textImageSplitStyles.functionality_media, textImageSplitStyles.functionality_media_supress_bottom_margin)}>
                <Image src={'/excel-to-python/mito_code_gen.png'} alt='Automate analysis with Mito' width={560} height={379} layout='responsive'/>
              </div>
            </div>
          </section>

        </main>
      </div>
      <Footer/>
    </>
  );
};

export default PageContent;

