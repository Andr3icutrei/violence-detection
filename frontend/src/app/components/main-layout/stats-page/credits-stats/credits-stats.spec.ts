import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CreditsStats } from './credits-stats';

describe('CreditsStats', () => {
  let component: CreditsStats;
  let fixture: ComponentFixture<CreditsStats>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CreditsStats],
    }).compileComponents();

    fixture = TestBed.createComponent(CreditsStats);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
