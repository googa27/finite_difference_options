import { useState } from "react";

export default function Home() {
  const [price, setPrice] = useState(null);

  async function fetchPrice() {
    const res = await fetch("http://localhost:8000/price", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        option_type: "Call",
        strike: 1,
        maturity: 1,
        s0: 1,
        rate: 0.05,
        sigma: 0.2,
      }),
    });
    const data = await res.json();
    setPrice(data.price);
  }

  return (
    <div>
      <h1>Option Pricing Demo</h1>
      <button onClick={fetchPrice}>Fetch Price</button>
      {price !== null && <p>Price: {price}</p>}
    </div>
  );
}
