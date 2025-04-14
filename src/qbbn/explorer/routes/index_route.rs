use rocket::response::content::RawHtml as Html;

use crate::qbbn::explorer::render_utils::render_app_body;


pub fn internal_index() -> Html<String> {
    let result = render_app_body("");
    Html(result.unwrap())
}