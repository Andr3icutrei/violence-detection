import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-portal-page',
  imports: [
    RouterOutlet,
  ],
  standalone: true,
  templateUrl: './portal-page.html',
  styleUrl: './portal-page.css',
})
export class PortalPage {
}
