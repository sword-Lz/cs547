$(document).ready(function () {
  $("button").click(function () {
    var searchTerm = $("input").val();
    $.ajax({
      url: "/getresults",
      method: "POST",
      data: JSON.stringify({ searchTerm: searchTerm }),
      contentType: "application/json",
      xhrFields: {
        withCredentials: true,
      },
      success: function (response) {
        $(".output-container").html(response);
      },
    });
  });
});
