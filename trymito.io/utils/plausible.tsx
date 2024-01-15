/* 
  In order to measure the effectiveness of the docs CTA, we add a custom css tag to all docs cta buttons 
  which plausible will track as a custom event. See here for more info: https://plausible.io/docs/custom-event-goals

  Note: Each of them __must__ start with `plausible-event-name=`
*/

// Install Mito DOCS CTA
export const PLAUSIBLE_INSTALL_DOCS_CTA_LOCATION_TITLE_CARD = 'plausible-event-name=install_docs_cta_pressed+location_title_card'
export const PLAUSIBLE_INSTALL_DOCS_CTA_LOCATION_FOOTER_CARD = 'plausible-event-name=install_docs_cta_pressed+location_footer_card'
export const PLAUSIBLE_INSTALL_DOCS_CTA_LOCATION_PLANS_OS = 'plausible-event-name=install_docs_cta_pressed+location_plans_os'
export const PLAUSIBLE_INSTALL_DOCS_CTA_LOCATION_HEADER = 'plausible-event-name=install_docs_cta_pressed+location_header'
export const PLAUSIBLE_INSTALL_DOCS_CTA_LOCATION_EXCEL_TO_PYTHON_GLOSSARY = 'plausible-event-name=install_docs_cta_pressed+location_excel_to_python_glossary'